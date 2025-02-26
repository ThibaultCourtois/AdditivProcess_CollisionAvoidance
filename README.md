# Notice technique du projet d'évitement de collisions lors de dépôts multi-axes

## Introduction

Cette notice présente le programme d'évitement de collisions développé pour le laboratoire COSMER. Ce programme permet de détecter et corriger automatiquement les collisions entre l'outil de dépôt et la pièce en cours de fabrication dans un processus de Wire Arc Additive Manufacturing (WAAM) multi-axes.

L'architecture modulaire du framework s'articule autour de plusieurs packages complémentaires :
- `TrajectoryDataManager` : Gestion des données de trajectoire
- `CollisionAvoidance` : Détection et correction des collisions
- `Visualizer` : Visualisation des trajectoires et des résultats
- `MetricsVisualizer` : Analyse des performances des corrections
- `main` : Script d'orchestration des opérations

## Configuration et exécution

### Prérequis

Le programme nécessite Python avec les bibliothèques suivantes :
- numpy
- pandas
- matplotlib

### Exécution du programme

Pour lancer le programme, exécutez le script principal `main.py` :

```bash
python main.py
```

## Paramétrage du script principal (main.py)

Le script `main.py` est le point d'entrée du programme et permet de paramétrer l'ensemble du processus de détection et de correction des collisions.

### Paramètres de fabrication

```python
# File paths for input trajectory and optimized output
INPUT_FILE = "Trajectoire_Schwarz.csv"
OUTPUT_FILE = "Trajectoire_Schwarz_Optimized.csv"

# Process geometric parameters
bead_width = 3.0       # Width of deposited material (mm)
bead_height = 1.95     # Height of deposited material (mm)
tool_radius = 12.5     # Radius of tool body (mm)
tool_length = 1000     # Overall length of tool (mm) (semi-infinite cylinder)
nozzle_length = 17.0   # Length of nozzle section (mm)
```

Ces paramètres définissent les caractéristiques géométriques essentielles :
- `INPUT_FILE` : Chemin du fichier de trajectoire d'entrée (format CSV)
- `OUTPUT_FILE` : Chemin où sera sauvegardée la trajectoire optimisée
- `bead_width` : Largeur du cordon de dépôt
- `bead_height` : Hauteur du cordon de dépôt
- `tool_radius` : Rayon de l'outil de dépôt
- `tool_length` : Longueur totale de l'outil
- `nozzle_length` : Longueur de la buse

### Paramètres de visualisation

```python
# Visualization resolution parameters
bead_discretization_points = 16    # Number of points for bead cross-section visualization
tool_discretization_points = 16    # Number of points for tool cylinder visualization

# Visualization range parameters
display_layers = [0, 79]           # Layer range to display
revolut_angle = 360                # Angular range for cylindrical parts
stride = 1                         # Step size for point sampling
```

Ces paramètres contrôlent la qualité et l'étendue de la visualisation :
- `bead_discretization_points` et `tool_discretization_points` : Résolution de la visualisation (plus la valeur est élevée, plus la visualisation sera précise)
- `display_layers` : Plage de couches à afficher [min, max]
- `revolut_angle` : Angle de révolution pour les pièces cylindriques (360° pour une visualisation complète)
- `stride` : Pas d'échantillonnage pour la visualisation des vecteurs

### Choix de la méthode de détection

Le programme propose deux méthodes de détection des collisions :

```python
# Perform initial collision detection
print("Detecting initial collisions...")
collision_points = collision_manager.detect_collisions_optimized()
#collision_points = collision_manager.detect_collisions_exhaustive()
```

Pour choisir la méthode souhaitée :
- Méthode optimisée (recommandée) : Utilisez `detect_collisions_optimized()`
- Méthode exhaustive (référence) : Commentez la ligne précédente et décommentez `detect_collisions_exhaustive()`

### Configuration de la visualisation initiale

```python
# Configure visualization options for initial analysis
initial_visualizer.setup_visualization(
    show_beads=False,          # Hide bead geometry for clarity
    low_res_bead=True,         # Use simplified bead representation
    show_vectors=False,        # Hide direction vectors
    show_tool=False,           # Hide tool geometry
    show_collisions=False,     # Highlight collision points
    show_collision_candidates=False,  # Hide potential collision points
    show_collision_bases=True        # Hide local coordinate systems
)
```

Les options de visualisation peuvent être activées/désactivées en modifiant les valeurs booléennes.

### Configuration de la visualisation finale

```python
# Configure visualization options for final analysis
optimized_visualizer.setup_visualization(
    show_beads=False,           # Hide bead geometry for clarity
    low_res_bead=True,          # Use simplified bead representation
    show_vectors=True,          # Show direction vectors
    show_tool=False,            # Hide tool geometry
    show_collisions=False,      # Highlight any remaining collisions
    show_collision_candidates=False,   # Hide potential collision points
    show_collision_bases=False         # Hide local coordinate systems
)
```

Vous pouvez configurer différemment la visualisation de la trajectoire optimisée.

## Utilisation des visualisations

### Options de visualisation disponibles

Le package `Visualizer` offre de nombreuses options configurables pour adapter l'affichage aux besoins d'analyse :

| Option | Paramètre | Description |
|--------|-----------|-------------|
| Chemin de dépôt | `display_layers` | Sélection des couches à afficher |
| | `stride` | Pas d'affichage des éléments (ex: une base locale tous les N points) |
| | `revolut_angle` | Secteur angulaire pour les pièces symétriques |
| Cordons | `show_beads` | Affichage des cordons de matière (ellipses) |
| | `low_res_bead` | Qualité d'affichage des ellipses (3 vs 6 rectangles par ellipse) |
| Outil | `show_tool` | Affichage de l'outil (cylindre + buse) |
| Vecteurs | `show_vectors` | Affichage des vecteurs des repères locaux |
| Collisions | `show_collisions` | Marquage des points de collision détectés |
| | `show_collision_candidates` | Affichage des points candidats à la collision |
| | `show_collision_bases` | Affichage des repères aux points de collision |

### Contrôles interactifs

En mode visualisation, plusieurs contrôles clavier sont disponibles :

| Touche | Action |
|--------|--------|
| '4' ou flèche gauche | Déplacement de l'outil au point précédent |
| '6' ou flèche droite | Déplacement de l'outil au point suivant |
| 'Ctrl + 4' ou 'Ctrl + flèche gauche' | Déplacement de l'outil de 10 points en arrière |
| 'Ctrl + 6' ou 'Ctrl + flèche droite' | Déplacement de l'outil de 10 points en avant |
| '8' ou 't' | Activer/désactiver l'affichage de l'outil |

Ces contrôles permettent d'explorer la trajectoire et d'analyser visuellement les résultats de la correction.

## Structure des données d'entrée/sortie

### Format du fichier de trajectoire

Le fichier de trajectoire d'entrée (et de sortie) est au format CSV avec les colonnes suivantes :

- X, Y, Z : Coordonnées du chemin de dépôt dans le repère pièce
- Bx, By, Bz : Coordonnées du vecteur de construction (Build Direction)
- Tx, Ty, Tz : Coordonnées du vecteur outil
- Extrusion : État d'extrusion (1 pour dépôt, 0 sinon)
- Autres colonnes pour la paramétrisation du dépôt (ne pas modifier)

## Exemples d'utilisation

### Exemple 1 : Analyse d'une trajectoire existante

```python
# Configuration des paramètres de base
INPUT_FILE = "Trajectoire_Schwarz.csv"
display_layers = [0, 60]  # Affichage des 60 premières couches

# Initialisation des gestionnaires
trajectory_manager = TrajectoryManager(INPUT_FILE)
collision_manager = CollisionAvoidance(
    trajectory_path=INPUT_FILE,
    bead_width=3.0,
    bead_height=1.95,
    tool_radius=12.5,
    tool_length=17.0,
)

# Génération des candidats à la collision
for layer_idx in range(len(trajectory_manager.layer_indices)):
    # Calcul des indices de début et fin pour la couche courante
    start_idx = trajectory_manager.layer_indices[layer_idx]
    end_idx = (trajectory_manager.layer_indices[layer_idx + 1]
               if layer_idx + 1 < len(trajectory_manager.layer_indices)
               else len(trajectory_manager.points))
    
    # Extraction des données spécifiques à la couche
    layer_points = trajectory_manager.points[start_idx:end_idx]
    layer_normals = trajectory_manager.n_vectors[start_idx:end_idx]
    layer_builds = trajectory_manager.b_vectors[start_idx:end_idx]
    
    # Génération des candidats à la collision pour la couche courante
    collision_manager.collision_candidates_generator.generate_collision_candidates_per_layer(
        points=layer_points,
        normal_vectors=layer_normals,
        build_vectors=layer_builds,
        layer_index=layer_idx,
    )

# Détection des collisions
collision_points = collision_manager.detect_collisions_optimized()

# Configuration du visualiseur pour l'affichage des collisions
visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=trajectory_manager,
    collision_manager=collision_manager,
    display_layers=display_layers,
    revolut_angle=360,
    stride=1
)

visualizer.setup_visualization(
    show_beads=False,
    show_vectors=False,
    show_tool=False,
    show_collisions=True,
    show_collision_bases=True
)

# Affichage des résultats
visualizer.create_figure()
visualizer.apply_layer_filter(layers=display_layers, angle_limit=360)
visualizer.visualize_trajectory()
visualizer.show()
```

### Exemple 2 : Correction et visualisation des résultats

```python
# Optimisation de la trajectoire
new_tool_vectors = collision_manager.process_trajectory()

# Sauvegarde de la trajectoire optimisée
trajectory_manager.save_modified_trajectory(new_tool_vectors, "Trajectoire_Optimized.csv")

# Affichage des métriques de correction
metrics_visualizer = MetricsVisualizer(collision_manager, trajectory_manager)
metrics_visualizer.visualize_tilt_angles_by_layer(threshold=5)
metrics_visualizer.visualize_corrected_points_3D(collision_manager, trajectory_manager)

# Visualisation de la trajectoire optimisée
optimized_trajectory = TrajectoryManager("Trajectoire_Optimized.csv")
optimized_visualizer = AdvancedTrajectoryVisualizer(
    trajectory_manager=optimized_trajectory,
    display_layers=display_layers,
    revolut_angle=360,
    stride=1
)

optimized_visualizer.setup_visualization(
    show_beads=False,
    show_vectors=True,
    show_tool=True,
    show_collisions=False
)

optimized_visualizer.create_figure()
optimized_visualizer.apply_layer_filter(layers=display_layers, angle_limit=360)
optimized_visualizer.visualize_trajectory()
optimized_visualizer.show()
```

## Conseils d'utilisation

1. **Performances** : La méthode de détection optimisée est recommandée pour les grandes trajectoires, offrant des gains significatifs en temps d'exécution.

2. **Visualisation** : Commencez par une visualisation simple (désactivez `show_beads` et `show_tool`) pour un rendu plus rapide, puis activez progressivement les options avancées selon vos besoins d'analyse.

3. **Analyse des corrections** : Utilisez `MetricsVisualizer` pour analyser la distribution des angles de correction et identifier les zones critiques de la trajectoire.

4. **Modification des paramètres géométriques** : Ajustez `bead_width`, `bead_height` et `tool_radius` selon les caractéristiques réelles de votre procédé de fabrication pour obtenir des résultats précis.

5. **Navigation interactive** : Utilisez les contrôles clavier pour explorer la trajectoire en détail et vérifier visuellement la qualité des corrections proposées.
