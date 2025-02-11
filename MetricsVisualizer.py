import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class MetricsVisualizer:
    def __init__(self, collision_manager, trajectory_manager):
        self.collision_avoidance = collision_manager  # Gestion collisions
        self.trajectory_manager = trajectory_manager  # Gestion trajectoire

    def visualize_tilt_angles_by_layer(self, selected_layers=None, threshold=5):
        """
        Affiche les angles de correction en fonction des couches.
        Permet d'afficher seulement certaines couches si besoin.
        
        :param selected_layers: Liste des indices de couches à afficher (None pour toutes)
        :param threshold: Seuil en degrés pour détecter une correction brusque
        """
        # Récupérer les données de correction (angles et indices)
        tilt_angles = self.collision_avoidance.get_tilt_angles()
        points_pb = self.collision_avoidance.get_points_pb()

        # Déterminer la couche de chaque point problématique
        layer_indices = self.trajectory_manager.layer_indices
        num_layers = len(layer_indices)

        # Dictionnaire pour stocker les corrections par couche
        layer_dict = {i: [] for i in range(num_layers)}

        for i, point_idx in enumerate(points_pb):
            for layer in range(num_layers):
                if point_idx < layer_indices[layer]:  # Trouver la première couche contenant le point
                    layer_dict[layer - 1].append((point_idx, tilt_angles[i]))
                    break
            else:
                layer_dict[num_layers - 1].append((point_idx, tilt_angles[i]))  # Dernière couche

        # a figure
        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(12, 6))

        for layer, data in layer_dict.items():
            if not data or (selected_layers and layer not in selected_layers):
                continue  # Ignorer les couches vides ou non sélectionnées
            
            indices, angles = zip(*data)  # Séparer les indices et les angles
            
            # Détection des corrections brusques
            brusque_corrections = np.where(np.abs(np.diff(angles)) > threshold)[0]

            # Tracé principal
            plt.plot(indices, angles, marker='o', linestyle='-', markersize=3, 
                     alpha=0.7, color=cmap(layer % 20), label=f'Couche {layer+1}')

            # Ajouter des marqueurs rouges aux transitions brusques
            if len(brusque_corrections) > 0:
                plt.scatter(np.array(indices)[brusque_corrections], np.array(angles)[brusque_corrections],
                            color='red', marker='x', s=50, label=f'Sauts brusques (Couche {layer+1})')

        plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Ligne 0°
        plt.xlabel('Index du point')
        plt.ylabel('Angle de correction (°)')
        plt.title(f'Angles de correction par couche (avec détection des changements supérieurs à {threshold} degrés)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Légende en dehors pour lisibilité
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()  # Optimise l'affichage
        plt.show()
        
    def visualize_tilt_angle_with_threshold(self, collision_manager, threshold=5):
        """Affiche les angles de correction avec zones de seuil"""
        
        tilt_angles = np.array(collision_manager.get_tilt_angles())
        points_pb = np.array(collision_manager.get_points_pb())
        
        plt.figure(figsize=(12, 6))
        plt.plot(points_pb, tilt_angles, label='Angles de correction', color='dodgerblue', alpha=0.7)
    
        # Marquer les zones où les angles dépassent le seuil
        too_large = np.abs(tilt_angles) > threshold
        plt.fill_between(points_pb, tilt_angles, where=too_large, color='red', alpha=0.3, label=f'Angle > {threshold}°')
    
        plt.xlabel("Index du point")
        plt.ylabel("Angle de correction (°)")
        plt.title(f"Correction des angles avec seuil de {threshold}°")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
        
        
    # def visualize_tilt_angles_polar(self, threshold=10):
    #     """
    #     Affiche une carte angulaire des corrections de tilt.
        
    #     :param threshold: Seuil en degrés pour détecter les corrections brusques
    #     """
    #     # Récupération des angles de correction
    #     tilt_angles = np.radians(self.collision_avoidance.get_tilt_angles())  # Conversion en radians
    #     points_pb = self.collision_avoidance.get_points_pb()  # Indices des points problématiques
        
    #     num_points = len(points_pb)
    #     radii = np.arange(num_points)  # Rayon croissant pour séparer les points
        
    #     # Détection des corrections brusques
    #     brusque_corrections = np.where(np.abs(np.diff(tilt_angles)) > np.radians(threshold))[0] + 1
        
    #     # Création du plot polaire
    #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    #     ax.set_theta_zero_location("N")  # Met 0° en haut
    #     ax.set_theta_direction(-1)  # Sens antihoraire
        
    #     # Tracé des corrections normales
    #     ax.scatter(tilt_angles, radii, c='blue', s=10, alpha=0.7, label="Corrections normales")
        
    #     # Tracé des corrections brusques en rouge
    #     if len(brusque_corrections) > 0:
    #         ax.scatter(tilt_angles[brusque_corrections], radii[brusque_corrections],
    #                    c='red', s=30, marker='x', label="Corrections brusques (> {}°)".format(threshold))
        
    #     # Personnalisation des axes
    #     ax.set_rticks([])  # Supprime les ticks radiaux
    #     ax.set_title("Carte angulaire des corrections d'angle", fontsize=14, fontweight='bold')
    #     ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))  # Déplace la légende à l'extérieur
    #     plt.show()
        
        
    # def visualize_tilt_heatmap(self, collision_manager, trajectory_manager):
    #     """Affiche une heatmap des angles de correction par couche"""
        
    #     tilt_angles = np.array(collision_manager.get_tilt_angles())
    #     points_pb = np.array(collision_manager.get_points_pb())  # Indices des points corrigés
        
    #     # Déterminer la couche de chaque point
    #     layer_indices = trajectory_manager.layer_indices
    #     num_layers = len(layer_indices)
    
    #     layers = np.zeros_like(points_pb)
    
    #     for i, point_idx in enumerate(points_pb):
    #         for layer in range(num_layers):
    #             if point_idx < layer_indices[layer]:
    #                 layers[i] = layer - 1
    #                 break
    #         else:
    #             layers[i] = num_layers - 1  # Dernière couche
    
    #     plt.figure(figsize=(12, 6))
    #     scatter = plt.scatter(points_pb, layers, c=tilt_angles, cmap='coolwarm', alpha=0.75, edgecolors='k')
        
    #     plt.colorbar(scatter, label="Angle de correction (°)")
    #     plt.xlabel("Index du point")
    #     plt.ylabel("Indice de la couche")
    #     plt.title("Heatmap des angles de correction par couche")
    #     plt.grid(True, linestyle="--", alpha=0.5)
    #     plt.show()
        
        
    # def visualize_tilt_angle_variation(self, collision_manager):
    #     """Affiche la variation des angles de correction entre chaque point"""
        
    #     tilt_angles = np.array(collision_manager.get_tilt_angles())
    #     points_pb = np.array(collision_manager.get_points_pb())
    
    #     # Calcul des variations entre chaque point
    #     variations = np.diff(tilt_angles)
        
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(points_pb[:-1], variations, marker='o', linestyle='-', markersize=3, color='purple', alpha=0.7)
    
    #     plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    #     plt.xlabel("Index du point")
    #     plt.ylabel("Variation d'angle (°)")
    #     plt.title("Variation des corrections d’angle entre chaque point")
    #     plt.grid(True, linestyle="--", alpha=0.5)
    #     plt.show()
        
        

    # def visualize_tilt_angles_3D(self, collision_manager, trajectory_manager):
    #     """Affiche un graphique 3D des corrections d'angle"""
        
    #     tilt_angles = np.array(collision_manager.get_tilt_angles())
    #     points_pb = np.array(collision_manager.get_points_pb())
    
    #     # Déterminer la couche de chaque point
    #     layer_indices = trajectory_manager.layer_indices
    #     num_layers = len(layer_indices)
    
    #     layers = np.zeros_like(points_pb)
    
    #     for i, point_idx in enumerate(points_pb):
    #         for layer in range(num_layers):
    #             if point_idx < layer_indices[layer]:
    #                 layers[i] = layer - 1
    #                 break
    #         else:
    #             layers[i] = num_layers - 1  # Dernière couche
    
    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(points_pb, layers, tilt_angles, c=tilt_angles, cmap='coolwarm', alpha=0.75)
    
    #     ax.set_xlabel("Index du point")
    #     ax.set_ylabel("Indice de la couche")
    #     ax.set_zlabel("Angle de correction (°)")
    #     ax.set_title("Correction des angles en 3D")
    
    #     plt.show()
        

    # def visualize_corrected_points_3D(self, collision_manager, trajectory_manager):
    #     """Affiche un graphique 3D des points corrigés avec les angles de correction"""
        
    #     tilt_angles = np.array(collision_manager.get_tilt_angles())
    #     points_pb = np.array(collision_manager.get_points_pb())
        
    #     corrected_points = trajectory_manager.points[points_pb]
        
    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     sc = ax.scatter(corrected_points[:, 0], corrected_points[:, 1], corrected_points[:, 2],
    #                     c=tilt_angles, cmap='coolwarm', s=50, alpha=0.7)
    
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Points corrigés (3D) avec angles de correction")
    
    #     plt.colorbar(sc, label="Angle de correction (°)")
    #     plt.show()

        

