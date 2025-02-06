import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
np.set_printoptions(threshold=np.inf)
np.seterr(all='ignore')  # Ignore les avertissements numpy pour plus de vitesse



"""
Système de visualisation optimisé utilisant les bases locales pré-calculées
"""

# --------------------------------
# Classe de gestion des bases locales
# --------------------------------
class LocalBasisManager:
    """Encapsule l'accès aux vecteurs de base depuis TrajectoryDataManager"""

    def __init__(self, trajectory_manager=None):
        self.trajectory_manager = trajectory_manager
        self.update_from_manager()

    def update_from_manager(self):
        """Récupère les vecteurs depuis le TrajectoryDataManager"""
        if self.trajectory_manager:
            self.t_vectors = self.trajectory_manager.t_vectors
            self.n_vectors = self.trajectory_manager.n_vectors
            self.b_vectors = self.trajectory_manager.b_vectors
            self.tool_vectors = self.trajectory_manager.tool_directions

    def get_local_basis(self, index):
        """Retourne la base locale (t,n,b) pour un point donné"""
        return (self.t_vectors[index],
                self.n_vectors[index],
                self.b_vectors[index],
                self.tool_vectors[index])

# --------------------------------
# Classes de visualisation géométrique
# --------------------------------
class GeometryVisualizer:
    def __init__(self, basis_manager):
        self.basis_manager = basis_manager

    def generate_bead_geometry_for_segment(self, segment_points, t_vectors, n_vectors, b_vectors, w_layer, h_layer, low_res_bead_bool):
        """
        Génère la géométrie des cordons de manière optimisée avec longueur adaptative
        """
        segment_length = len(segment_points)
        if segment_length < 2:
            return None

        if low_res_bead_bool:
            # Réduction du nombre de points pour les performances
            n_section_points = 4
        else :
            n_section_points = 8

        # Pour chaque point du segment, générer une demi-ellipse
        all_sections = []

        for i in range(segment_length):
            point = segment_points[i]
            t = t_vectors[i] / np.linalg.norm(t_vectors[i])
            n = n_vectors[i] / np.linalg.norm(n_vectors[i])
            b = b_vectors[i] / np.linalg.norm(b_vectors[i])

            # Calcul de la longueur adaptative basée sur les points adjacents
            if i > 0:
                prev_dist = np.linalg.norm(segment_points[i] - segment_points[i - 1])
            else:
                prev_dist = 0

            if i < segment_length - 1:
                next_dist = np.linalg.norm(segment_points[i] - segment_points[i + 1])
            else:
                next_dist = 0

            # Longueur adaptative avec limites
            L_seg = np.mean([d for d in [prev_dist, next_dist] if d > 0])
            L_seg = np.clip(L_seg, 0.2, 1.0)  # Limites min/max en mm

            # Points du rectangle de base (optimisé)
            n_vals = np.linspace(-w_layer / 2, w_layer / 2, n_section_points)
            section_points = []

            # Construction optimisée des points
            for n_val in n_vals:
                # Points avant
                section_points.append(point + n_val * n + L_seg / 2 * t)
            # Points arrière (en ordre inverse)
            for n_val in reversed(n_vals):
                section_points.append(point + n_val * n - L_seg / 2 * t)

            section_points = np.array(section_points)

            # Déformation en demi-ellipse (calcul optimisé)
            distances = np.abs(section_points - point)
            max_distance = np.max(np.linalg.norm(distances, axis=1))
            height_factors = np.cos(np.linalg.norm(distances, axis=1) / max_distance * np.pi / 2)
            height_variation = (h_layer / 2) * height_factors[:, np.newaxis] * b

            section = section_points + height_variation
            all_sections.append(section)

        return all_sections

    def plot_bead_sections(self, ax, sections):
        """
        Version corrigée du tracé des sections avec gestion appropriée des vertices
        """
        aluminum_color = '#D3D3D3'

        for section in sections:
            n_points = len(section)
            mid_point = n_points // 2

            # Création optimisée des surfaces
            for i in range(mid_point - 1):
                # Création d'un quadrilatère avec les 4 points dans le bon format
                # Convert numpy arrays to lists to ensure proper formatting
                quad = [
                    section[i].tolist(),
                    section[i + 1].tolist(),
                    section[-(i + 2)].tolist(),
                    section[-(i + 1)].tolist(),
                ]

                # Créer la collection avec un seul quad à la fois
                poly = Poly3DCollection([quad])
                poly.set_facecolor(aluminum_color)
                poly.set_edgecolor('black')
                poly.set_linewidth(0.1)
                poly.set_rasterized(True)
                poly.set_zsort('min')
                ax.add_collection3d(poly)

    def generate_tool_geometry(self, point, tool_dir):
        """Génère la géométrie 3D de l'outil de façon robuste"""
        # Normalisation de la direction
        tool_dir = tool_dir / np.linalg.norm(tool_dir)

        # Construction d'une base orthonormale robuste
        z_axis = tool_dir
        if abs(z_axis[0]) < 1e-6 and abs(z_axis[1]) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Génération des points du cylindre
        theta = np.linspace(0, 2 * np.pi, self.n_tool_points)
        circle_points = np.column_stack((
            self.tool_radius * np.cos(theta),
            self.tool_radius * np.sin(theta),
            np.zeros_like(theta)
        ))

        # Application de la transformation
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        nozzle_end = point + self.nozzle_length * tool_dir

        # Création des cercles de base et sommet
        base_circle = np.dot(circle_points, rotation_matrix.T) + nozzle_end
        top_circle = base_circle + self.tool_height * tool_dir

        return {
            'base': base_circle,
            'top': top_circle,
            'nozzle_start': point,
            'nozzle_end': nozzle_end
        }

    def set_parameters(self, bead_width=None, bead_height=None,
                       tool_radius=None, tool_length=None, n_bead_points = None, nozzle_length = None, n_tool_points = None):
        """Configure les paramètres géométriques"""
        # Paramètres de l'outil
        if tool_radius:
            self.tool_radius = tool_radius
        if tool_length:
            self.tool_height = tool_length
        if bead_width:
            self.bead_width = bead_width
        if bead_height:
            self.bead_height = bead_height
        if n_bead_points:
            self.n_bead_points = n_bead_points
        if nozzle_length:
            self.nozzle_length = nozzle_length
        if n_tool_points:
            self.n_tool_points = n_tool_points

# --------------------------------
# Classe principale de visualisation
# --------------------------------
class AdvancedTrajectoryVisualizer:
    """Gestionnaire principal de la visualisation"""

    def __init__(self, trajectory_manager=None, collision_manager=None, display_layers=None,
                 revolut_angle=360, stride=1):
        self.trajectory_manager = trajectory_manager
        self.collision_manager = collision_manager
        self.basis_manager = LocalBasisManager(trajectory_manager)
        self.geometry_visualizer = GeometryVisualizer(self.basis_manager)
        self.collision_points = None
        if collision_manager:
            self.collision_points = collision_manager.detect_collisions_optimized()
            #self.collision_points = collision_manager.detect_collisions_exhaustive()

        # Paramètres de visualisation
        self.display_layers = display_layers or []
        self.revolut_angle = revolut_angle
        self.stride = stride

        # États de visualisation
        self.show_beads = False
        self.show_tool = False
        self.show_vectors = False
        self.show_collisions = False
        self.low_res_bead = True
        self.show_collision_candidates = False
        self.show_collision_bases = False

        # Navigation
        self.current_point = 0
        self.visible_points = None
        self.visible_indices = None

        # Figure matplotlib
        self.fig = None
        self.ax = None

        self.tool_artists = []



    def setup_visualization(self, show_beads=False, low_res_bead=True, show_tool=False,
                          show_vectors=False, show_collisions=False, show_collision_candidates=False, show_collision_bases=False):
        """Configure les options de visualisation"""
        self.show_beads = show_beads
        self.show_tool = show_tool
        self.show_vectors = show_vectors
        self.show_collisions = show_collisions
        self.low_res_bead = low_res_bead
        self.show_collision_candidates = show_collision_candidates
        self.show_collision_bases = show_collision_bases

    def apply_layer_filter(self, layers=None, angle_limit=None):
        """Filtre les points avec prise en compte de l'extrusion."""
        if not layers:
            return

        # Récupération des indices de début et fin
        start_idx = self.trajectory_manager.layer_indices[min(layers)]
        end_idx = (self.trajectory_manager.layer_indices[max(layers) + 1]
                   if max(layers) + 1 < len(self.trajectory_manager.layer_indices)
                   else len(self.trajectory_manager.points))

        # Sélection des points et vecteurs
        selected_points = self.trajectory_manager.points[start_idx:end_idx]
        selected_t = self.basis_manager.t_vectors[start_idx:end_idx]
        selected_n = self.basis_manager.n_vectors[start_idx:end_idx]
        selected_b = self.basis_manager.b_vectors[start_idx:end_idx]
        selected_tool = self.trajectory_manager.tool_directions[start_idx:end_idx]
        selected_extrusion = self.trajectory_manager.extrusion[start_idx:end_idx]

        # Application du filtre par angle
        if angle_limit is not None and angle_limit != 360:
            angles = np.degrees(np.arctan2(selected_points[:, 1], selected_points[:, 0]))
            angles = np.where(angles < 0, angles + 360, angles)
            angle_mask = angles <= angle_limit

            selected_points = selected_points[angle_mask]
            selected_t = selected_t[angle_mask]
            selected_n = selected_n[angle_mask]
            selected_b = selected_b[angle_mask]
            selected_tool = selected_tool[angle_mask]
            selected_extrusion = selected_extrusion[angle_mask]

            # Stocker les indices filtrés et visibles
            global_indices = np.arange(start_idx, end_idx)[angle_mask]
        else:
            global_indices = np.arange(start_idx, end_idx)

        self.visible_points = selected_points
        self.visible_t_vector = selected_t
        self.visible_n_vector = selected_n
        self.visible_b_vector = selected_b
        self.visible_tool_vector = selected_tool
        self.visible_extrusion = selected_extrusion

        # Mapping des indices globaux vers indices visibles
        self.global_to_visible_indices = {global_idx: local_idx for local_idx, global_idx in enumerate(global_indices)}

        # Stocker les indices visibles
        self.visible_indices = global_indices

    def create_figure(self):
        """Crée et configure la figure matplotlib avec optimisations avancées"""
        self.fig = plt.figure(figsize=(12, 8), dpi=100)  # Réduit la résolution
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Optimisations de rendu
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Désactive complètement les panneaux de fond
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Désactive les grilles pour améliorer les performances
        self.ax.grid(False)

        # Désactive le calcul automatique des limites
        self.ax.autoscale(enable=False)

        # Utilise une projection orthographique (plus rapide)
        self.ax.set_proj_type('ortho')

        # Désactive l'anti-aliasing
        self.fig.set_facecolor('white')
        self.ax.set_facecolor('white')

        # Connecter les événements
        self.fig.canvas.mpl_connect('key_press_event', self.handle_keyboard)

    def visualize_trajectory(self):
        """Visualisation de la trajectoire avec gestion des sauts par extrusion"""
        if not self.fig:
            self.create_figure()

        # Application des filtres
        self.apply_layer_filter(self.display_layers, self.revolut_angle)

        # Séparation en segments basée sur l'extrusion
        points = self.visible_points
        segments = []
        current_segment = []

        # Index de début pour le filtrage
        start_idx = self.trajectory_manager.layer_indices[min(self.display_layers)]

        for i in range(len(points)):
            if self.visible_extrusion[i] == 1:  # Point avec extrusion
                if len(current_segment) == 0 or self.visible_extrusion[i - 1] == 1:
                    current_segment.append(points[i])
                else:
                    # Nouveau segment si on vient d'un point sans extrusion
                    if len(current_segment) > 1:
                        segments.append(np.array(current_segment))
                    current_segment = [points[i]]
            else:
                # Fermeture du segment en cours si existe
                if len(current_segment) > 1:
                    segments.append(np.array(current_segment))
                current_segment = []

        # Ajout du dernier segment
        if len(current_segment) > 1:
            segments.append(np.array(current_segment))

        # Tracé des segments
        for i, segment in enumerate(segments):
            self.ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                         'k-', label='Trajectory' if i == 0 else "")

        # Ajout des éléments optionnels
        if self.show_vectors:
            self._draw_vectors()
        if self.show_beads:
            self._draw_beads()
        if self.show_collisions:
            self._draw_collisions()
        if self.show_tool:
            self.update_tool_visualization()
        if self.show_collision_candidates:
            self.visualize_collision_candidates(
                self.visible_points,
                self.visible_n_vector,
                self.visible_b_vector
            )
        if self.show_collision_bases:
            self._draw_collision_bases()
        # Configuration des limites et aspect
        self._setup_plot_limits()

    def update_tool_visualization(self):
        """Met à jour la visualisation de l'outil de façon robuste"""
        # Nettoyage des anciens éléments
        for artist in self.tool_artists:
            artist.remove()
        self.tool_artists.clear()

        if not self.show_tool or self.current_point is None:
            return

        # Récupération des données géométriques
        point = self.visible_points[self.current_point]
        tool_dir = self.visible_tool_vector[self.current_point]

        # Génération de la géométrie
        geom = self.geometry_visualizer.generate_tool_geometry(point, tool_dir)

        # Création du cylindre
        for circle in [geom['base'], geom['top']]:
            # Fermeture du cercle
            circle_closed = np.vstack([circle, circle[0]])
            line = self.ax.plot(circle_closed[:, 0],
                                circle_closed[:, 1],
                                circle_closed[:, 2],
                                'k-', linewidth=0.5)[0]
            self.tool_artists.append(line)

        # Surface du cylindre
        cylinder_x = np.vstack((geom['base'][:, 0], geom['top'][:, 0]))
        cylinder_y = np.vstack((geom['base'][:, 1], geom['top'][:, 1]))
        cylinder_z = np.vstack((geom['base'][:, 2], geom['top'][:, 2]))

        surf = self.ax.plot_surface(cylinder_x, cylinder_y, cylinder_z,
                                    color='gray', alpha=0.3)
        self.tool_artists.append(surf)

        # Ligne de la buse
        line = self.ax.plot([geom['nozzle_start'][0], geom['nozzle_end'][0]],
                            [geom['nozzle_start'][1], geom['nozzle_end'][1]],
                            [geom['nozzle_start'][2], geom['nozzle_end'][2]],
                            'k-', linewidth=2.0)[0]
        self.tool_artists.append(line)

        self.fig.canvas.draw()

    def _draw_beads(self):
        """Dessine les cordons en utilisant l'approche par segments"""
        if len(self.visible_indices) < 1:
            return

        # Paramètres du cordon
        w_layer = self.collision_manager.collision_candidates_generator.w_layer
        h_layer = self.collision_manager.collision_candidates_generator.h_layer

        # Segmentation de la trajectoire
        points = self.visible_points
        segments = []
        current_segment = []
        current_t = []
        current_n = []
        current_b = []

        for i in range(len(points)):
            if self.visible_extrusion[i] == 1:
                if len(current_segment) == 0 or self.visible_extrusion[i - 1] == 1:
                    # Ajout au segment courant
                    current_segment.append(points[i])
                    current_t.append(self.visible_t_vector[i])
                    current_n.append(self.visible_n_vector[i])
                    current_b.append(self.visible_b_vector[i])
                else:
                    # Nouveau segment
                    if len(current_segment) > 1:
                        segments.append({
                            'points': np.array(current_segment),
                            't_vectors': np.array(current_t),
                            'n_vectors': np.array(current_n),
                            'b_vectors': np.array(current_b)
                        })
                    current_segment = [points[i]]
                    current_t = [self.visible_t_vector[i]]
                    current_n = [self.visible_n_vector[i]]
                    current_b = [self.visible_b_vector[i]]
            else:
                # Fin du segment courant
                if len(current_segment) > 1:
                    segments.append({
                        'points': np.array(current_segment),
                        't_vectors': np.array(current_t),
                        'n_vectors': np.array(current_n),
                        'b_vectors': np.array(current_b)
                    })
                current_segment = []
                current_t = []
                current_n = []
                current_b = []

        # Ajout du dernier segment si nécessaire
        if len(current_segment) > 1:
            segments.append({
                'points': np.array(current_segment),
                't_vectors': np.array(current_t),
                'n_vectors': np.array(current_n),
                'b_vectors': np.array(current_b)
            })

        # Tracé des segments
        for segment in segments:
            sections = self.geometry_visualizer.generate_bead_geometry_for_segment(
                segment['points'],
                segment['t_vectors'],
                segment['n_vectors'],
                segment['b_vectors'],
                w_layer,
                h_layer,
                self.low_res_bead
            )
            if sections:
                self.geometry_visualizer.plot_bead_sections(self.ax, sections)

    def _draw_vectors(self):
        """Dessine les vecteurs avec gestion plus stricte des vecteurs tool"""
        plot_points = []
        plot_vectors = []
        colors = []

        for i in range(0, len(self.visible_points), self.stride):
            point = self.visible_points[i]
            t = self.visible_t_vector[i]
            n = self.visible_n_vector[i]
            b = self.visible_b_vector[i]
            tool = self.visible_tool_vector[i]

            # Ajout des vecteurs de base
            plot_points.extend([point, point, point])
            plot_vectors.extend([t, n, b])
            colors.extend(['blue', 'green', 'red'])

            # Vérification plus stricte pour le vecteur tool
            norm_b = b / np.linalg.norm(b)
            norm_tool = tool / np.linalg.norm(tool)
            angle_diff = np.arccos(np.clip(np.dot(norm_b, norm_tool), -1.0, 1.0))

            # N'afficher le vecteur tool que si l'angle est significatif (>5 degrés)
            if np.degrees(angle_diff) > 5:
                plot_points.append(point)
                plot_vectors.append(tool)
                colors.append('orange')

        # Tracé vectorisé
        plot_points = np.array(plot_points)
        plot_vectors = np.array(plot_vectors)

        if len(plot_points) > 0:
            self.ax.quiver(
                plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                plot_vectors[:, 0], plot_vectors[:, 1], plot_vectors[:, 2],
                colors=colors, normalize=True
            )

            # Légende avec uniquement les vecteurs tool significatifs
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', label='Tangent direction'),
                Line2D([0], [0], color='green', label='Normal direction'),
                Line2D([0], [0], color='red', label='Build direction')
            ]
            if 'orange' in colors:
                legend_elements.append(Line2D([0], [0], color='orange', label='Tool direction'))

            self.ax.legend(handles=legend_elements)

    def _draw_collisions(self):
        """Dessine les points de collision avec une taille réduite"""
        if self.collision_manager and self.collision_points is not None:
            # Ne garder que les collisions des points visibles
            collision_indices = np.where(self.collision_points)[0]
            mask = np.isin(collision_indices, self.visible_indices)

            if np.any(mask):
                collision_points = self.trajectory_manager.points[collision_indices[mask]]
                self.ax.scatter(collision_points[:, 0],
                                collision_points[:, 1],
                                collision_points[:, 2],
                                c='red', marker='x', s=3,  # Taille réduite des marqueurs
                                label='Collisions')

    def _draw_collision_bases(self):
        """Dessine les bases locales aux points de collision."""
        if self.collision_manager and self.collision_points is not None:
            # Obtenir les indices globaux des points en collision
            collision_indices = self.collision_manager.get_collision_indices()

            # Mapper vers les indices visibles
            visible_collision_indices = [
                self.global_to_visible_indices[global_idx]
                for global_idx in collision_indices
                if global_idx in self.global_to_visible_indices
            ]

            if not visible_collision_indices:
                print("Aucun point de collision visible après filtrage.")
                return

            plot_points = []
            plot_vectors = []
            colors = []

            for idx in visible_collision_indices:
                point = self.visible_points[idx]
                t = self.visible_t_vector[idx]
                n = self.visible_n_vector[idx]
                b = self.visible_b_vector[idx]

                # Normaliser les vecteurs
                t = t / np.linalg.norm(t)
                n = n / np.linalg.norm(n)
                b = b / np.linalg.norm(b)

                # Ajouter les vecteurs de base
                plot_points.extend([point, point, point])
                plot_vectors.extend([t, n, b])
                colors.extend(['blue', 'green', 'red'])

            if len(plot_points) > 0:
                plot_points = np.array(plot_points)
                plot_vectors = np.array(plot_vectors)

                self.ax.quiver(
                    plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
                    plot_vectors[:, 0], plot_vectors[:, 1], plot_vectors[:, 2],
                    colors=colors, normalize=True,
                )

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', label='Tangent direction (t)'),
                Line2D([0], [0], color='green', label='Normal direction (n)'),
                Line2D([0], [0], color='red', label='Build direction (b)'),
            ]
            self.ax.legend(handles=legend_elements)
            print(f"Visualisation des bases locales pour {len(visible_collision_indices)} points de collision.")

    def _draw_tool(self):
        """Dessine l'outil avec correction pour la visualisation"""
        if self.show_tool and self.current_point is not None:
            if 0 <= self.current_point < len(self.visible_indices):
                point_idx = self.visible_indices[self.current_point]
                point = self.visible_points[self.current_point]

                # Générer et afficher la géométrie de l'outil
                geom = self.geometry_visualizer.generate_tool_geometry(point, point_idx)

                # Reshape pour la visualisation surface
                n_points = len(geom)
                surf_x = geom[:, 0].reshape(2, -1)
                surf_y = geom[:, 1].reshape(2, -1)
                surf_z = geom[:, 2].reshape(2, -1)

                self.ax.plot_surface(surf_x, surf_y, surf_z,
                                     color='gray', alpha=0.5)

    def _setup_plot_limits(self):
        """Configure les limites de l'affichage"""
        if len(self.visible_points) > 0:
            # Calcul des dimensions
            x_range = np.ptp(self.visible_points[:,0])
            y_range = np.ptp(self.visible_points[:,1])
            z_range = np.ptp(self.visible_points[:,2])

            max_range = max(x_range, y_range, z_range)
            center = np.mean(self.visible_points, axis=0)

            # Application des limites
            self.ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            self.ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
            self.ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

            # Force un aspect ratio égal
            self.ax.set_box_aspect([1,1,1])

    def visualize_collision_candidates(self, points, normal_vectors, build_vectors):
        """
        Visualize collision candidate points using filtered data
        """
        if len(points) == 0:
            return

        # Utiliser les dimensions depuis le collision manager
        w_layer = self.collision_manager.collision_candidates_generator.w_layer
        h_layer = self.collision_manager.collision_candidates_generator.h_layer

        # Points gauche et droite pour tous les points (avec stride)
        strided_points = points[::self.stride]
        strided_normal = normal_vectors[::self.stride]
        strided_build = build_vectors[::self.stride]

        # Calcul des points candidats
        left_points = strided_points + (w_layer / 2) * strided_normal
        right_points = strided_points - (w_layer / 2) * strided_normal

        # Affichage des points candidats latéraux
        all_points = np.vstack([left_points, right_points])
        self.ax.scatter(all_points[:, 0],
                        all_points[:, 1],
                        all_points[:, 2],
                        color='darkcyan',
                        marker='o',
                        s=1,
                        label='Collision candidates')

        # Points supérieurs pour la dernière couche seulement
        if self.display_layers:
            max_layer = max(self.display_layers)
            start_idx = self.trajectory_manager.layer_indices[max_layer - 1] if max_layer > 0 else 0
            end_idx = self.trajectory_manager.layer_indices[max_layer]
            is_last_layer = np.arange(len(strided_points))[::self.stride]
            last_layer_points = strided_points[is_last_layer]
            last_layer_build = strided_build[is_last_layer]
            top_points = last_layer_points + h_layer * last_layer_build

            # Affichage des points supérieurs
            self.ax.scatter(top_points[:, 0],
                            top_points[:, 1],
                            top_points[:, 2],
                            color='darkcyan',
                            marker='o',
                            s=1)

    def handle_keyboard(self, event):
        """Gestion des événements clavier"""
        if not hasattr(self, 'current_point'):
            self.current_point = 0

        if event.key in ['left', '4']:
            self.current_point = max(0, self.current_point - 1)
            self.update_tool_visualization()
        elif event.key in ['right', '6']:
            self.current_point = min(len(self.visible_points) - 1, self.current_point + 1)
            self.update_tool_visualization()
        elif event.key in ['t', '8']:
            self.show_tool = not self.show_tool
            self.update_tool_visualization()
        elif event.key.startswith('ctrl+'):
            step = 10
            if 'left' in event.key or '4' in event.key:
                self.current_point = max(0, self.current_point - step)
            elif 'right' in event.key or '6' in event.key:
                self.current_point = min(len(self.visible_points) - 1, self.current_point + step)
            self.update_tool_visualization()

    def update_visualization(self):
        """Met à jour la visualisation après changement d'état"""
        self.ax.clear()
        self.visualize_trajectory()
        plt.draw()

    def show(self):
        """Affiche la visualisation"""
        if not self.fig:
            self.create_figure()
        self.visualize_trajectory()
        plt.show()