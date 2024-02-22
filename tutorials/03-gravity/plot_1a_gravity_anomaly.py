"""
Forward Simulation of Gravity Anomaly Data on a Tensor Mesh
===========================================================

Here we use the module *SimPEG.potential_fields.gravity* to predict gravity
anomaly data for a synthetic density contrast model. The simulation is
carried out on a tensor mesh. For this tutorial, we focus on the following:

    - How to create gravity surveys
    - How to predict gravity anomaly data for a density contrast model
    - How to include surface topography
    - The units of the density contrast model and resulting data


"""

#########################################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.interpolate import LinearNDInterpolator  #fornisce funzionalità per l'interpolazione lineare in più dimensioni.
import matplotlib as mpl 
import matplotlib.pyplot as plt
import os   #È una libreria Python che fornisce funzionalità per l'interazione con il sistema operativo. Viene utilizzato qui per controllare il salvataggio dei risultati.

from discretize import TensorMesh   #fornisce funzionalità per la creazione e la manipolazione di mesh tensore utilizzate per risolvere problemi di equazioni differenziali alle derivate parziali (PDE) e altre applicazioni numeriche.
from discretize.utils import mkvc, active_from_xyz  #fornisce una funzione per appiattire una matrice in un vettore e una funzione per ottenere un array booleano che indica le celle attive (sotto la superficie) da un insieme di punti xyz.

from SimPEG.utils import plot2Ddata, model_builder # fornisce funzionalità per la visualizzazione dei dati in due dimensioni e la costruzione di modelli geofisici utilizzati nelle simulazioni.
from SimPEG import maps #fornisce funzionalità per la creazione di mappe utilizzate per la trasformazione dei modelli.
from SimPEG.potential_fields import gravity #fornisce funzionalità per la simulazione della gravità e il calcolo dell'anomalia gravitazionale.

save_output = False #viene utilizzata per controllare se salvare o meno i risultati della simulazione. Se impostata su True, i risultati verranno salvati, altrimenti no.

# sphinx_gallery_thumbnail_number = 2

#############################################
# Defining Topography
# -------------------
#
# Surface topography is defined as an (N, 3) numpy array. We create it here but
# the topography could also be loaded from a file.
#

[x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))  #Viene creato un meshgrid di coordinate x e y utilizzando np.linspace per generare valori equamente spaziati all'interno di un intervallo specificato.
z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2) #Viene definita una superficie z_topo basata su una funzione esponenziale di x e y.
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)   #Le coordinate x, y e z della topografia sono appiattite utilizzando la funzione mkvc per ottenere un array 1D per ciascuna coordinata.
topo_xyz = np.c_[x_topo, y_topo, z_topo]    #Le coordinate della topografia sono concatenate per formare un array 2D topo_xyz in cui ogni riga rappresenta le coordinate (x, y, z) di un punto della topografia.


#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for the forward simulation. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations, and a list of field components
# which are to be measured.
#

# Define the observation locations as an (N, 3) numpy array or load them.
x = np.linspace(-80.0, 80.0, 17) #Genera un vettore di coordinate x che varia da -80.0 a 80.0, con 17 punti equispaziati.
y = np.linspace(-80.0, 80.0, 17)    #Genera un vettore di coordinate y simile al punto precedente.
x, y = np.meshgrid(x, y)    # Crea una griglia bidimensionale di coordinate x e y utilizzando la funzione meshgrid.
x, y = mkvc(x.T), mkvc(y.T) # Converte le matrici delle coordinate x e y in vettori colonna usando la funzione mkvc.
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)    #Crea un'interpolazione lineare tra le coordinate x e y della topografia (concatenate in un'unica matrice con np.c_) e le corrispondenti coordinate z. Questo interpolatore verrà utilizzato per calcolare le coordinate z delle osservazioni basate sulla topografia.
z = fun_interp(np.c_[x, y]) + 5.0   #Calcola le coordinate z delle osservazioni utilizzando l'interpolatore creato in precedenza e aggiunge un offset di 5.0 metri. Questo offset serve per posizionare le osservazioni sopra la superficie topografica.
receiver_locations = np.c_[x, y, z] #Combina le coordinate x, y e z delle osservazioni in una matrice tridimensionale, dove ogni riga rappresenta le coordinate di un singolo punto di osservazione.

# Define the component(s) of the field we want to simulate as strings within
# a list. Here we simulate only the vertical component of gravity anomaly.
components = ["gz"]

# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_list = gravity.receivers.Point(receiver_locations, components=components)  #crea un elenco di ricevitori per la simulazione della gravità, ossia una lista di oggetti Point, ognuno dei quali rappresenta un ricevitore per la simulazione della gravità, con le rispettive posizioni e il componente del campo da misurare.

receiver_list = [receiver_list]

# Defining the source field.
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Defining the survey
survey = gravity.survey.Survey(source_field)    #definisce il campo sorgente per la simulazione della gravità


#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to predict gravity anomaly
# data.
#

dh = 5.0    #definisce la distanza tra i nodi della mesh
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]    #Questa e le seguenti variabili definiscono la geometria della mesh lungo gli assi x, y e z rispettivamente. Ogni tupla ha tre elementi: la distanza tra i nodi, il numero di nodi nell'intervallo e un fattore di stretching opzionale
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")  #Questa riga crea effettivamente la mesh tensoriale utilizzando gli intervalli definiti lungo gli assi x, y e z. L'argomento opzionale "CCN" specifica il tipo di mesh che viene creata

########################################################
# Density Contrast Model and Mapping on Tensor Mesh
# -------------------------------------------------
#
# Here, we create the density contrast model that will be used to predict
# gravity anomaly data and the mapping from the model to the mesh. The model
# consists of a less dense block and a more dense sphere.
#

# Define density contrast values for each unit in g/cc
background_density = 0.0
block_density = -0.2
sphere_density = 0.2

# Find the indices for the active mesh cells (e.g. cells below surface)
ind_active = active_from_xyz(mesh, topo_xyz)    
"""La funzione active_from_xyz prende due argomenti:
mesh: La mesh sulla quale verranno calcolati gli indici delle celle attive.
topo_xyz: Le coordinate xyz della superficie topografica.
Restituisce un array booleano che indica quali celle della mesh sono attive. Gli indici delle celle attive sono quelli per i quali il valore nell'array booleano è True."""

# Define mapping from model to active cells. The model consists of a value for
# each cell below the Earth's surface.
nC = int(ind_active.sum())  #viene calcolato il numero totale di celle attive (nC) contando il numero di valori True nell'array booleano ind_active, che indica le celle attive.
model_map = maps.IdentityMap(nP=nC) # mappa ciascun valore del modello direttamente a una cella attiva corrispondente senza alcuna trasformazione o riduzione delle dimensioni. Il parametro nP viene impostato a nC, indicando che il modello ha una dimensione pari al numero di celle attive.

# Define model. Models in SimPEG are vector arrays.
model = background_density * np.ones(nC)    #crea un array numpy in cui ogni elemento ha il valore di background_density. Questo array rappresenta il modello per la simulazione della densità di fondo in ciascuna cella attiva della mesh.

# You could find the indicies of specific cells within the model and change their
# value to add structures.
ind_block = (   #ind_block è un array booleano che indica quali celle attive della mesh corrispondono alla regione specificata
    (mesh.gridCC[ind_active, 0] > -50.0)    #Seleziona le celle attive della mesh (quelle sotto la superficie) che hanno una coordinata x maggiore di -50.0
    & (mesh.gridCC[ind_active, 0] < -20.0)  #Effettua un'ulteriore selezione sulle celle attive, richiedendo che la coordinata x sia anche minore di -20.0.
    & (mesh.gridCC[ind_active, 1] > -15.0)  #Applica una condizione simile alla coordinata y delle celle attive, richiedendo che sia maggiore di -15.0.
    & (mesh.gridCC[ind_active, 1] < 15.0)   #Impone un limite superiore sulla coordinata y delle celle attive, richiedendo che sia minore di 15.0.
    & (mesh.gridCC[ind_active, 2] > -50.0)  #Effettua una selezione sulla coordinata z delle celle attive, richiedendo che sia maggiore di -50.0.
    & (mesh.gridCC[ind_active, 2] < -30.0)  #Impone un limite superiore sulla coordinata z delle celle attive, richiedendo che sia minore di -30.0.
)
model[ind_block] = block_density    #Assegna il valore block_density alle celle della mesh che soddisfano tutte le condizioni specificate in ind_block.

# You can also use SimPEG utilities to add structures to the model more concisely
ind_sphere = model_builder.get_indices_sphere(  #ottenere gli indici delle celle attive della mesh che si trovano all'interno di una sfera con centro in (35.0, 0.0, -40.0) e raggio 15.0. Questa funzione restituisce un array booleano che indica quali celle della mesh si trovano all'interno della sfera.
    np.r_[35.0, 0.0, -40.0], 15.0, mesh.gridCC
)
ind_sphere = ind_sphere[ind_active] #Questa riga filtra gli indici ottenuti nella riga precedente, mantenendo solo quelli corrispondenti alle celle attive della mesh, come indicato dall'array booleano ind_active.
model[ind_sphere] = sphere_density  #Questa riga aggiorna il valore di densità del modello per le celle che corrispondono a True in ind_sphere, impostandole a sphere_density. Questo passaggio consente di aggiungere una struttura sferica di densità al modello nelle celle selezionate.

# Plot Density Contrast Model
fig = plt.figure(figsize=(9, 4))    #crea una nuova figura matplotlib con una larghezza di 9 pollici e un'altezza di 4 pollici.
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan) #mappatura delle celle attive della mesh. Le celle non attive sono impostate su np.nan, in modo che non vengano visualizzate nel plot.

ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78]) #aggiunge un'area degli assi alla figura. Gli argomenti [0.1, 0.12, 0.73, 0.78] specificano le posizioni e le dimensioni dell'area degli assi all'interno della figura.
mesh.plot_slice(    # traccia una sezione del modello di densità lungo l'asse Y della mesh
    plotting_map * model,   #viene utilizzato per mascherare le celle non attive impostando i loro valori su np.nan.
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    clim=(np.min(model), np.max(model)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m") # Imposta il titolo del plot.
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])    #aggiunge un'area degli assi per la barra dei colori alla destra del plot principale.
norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model)) # Normalizza i valori del modello per essere utilizzati nella barra dei colori.
cbar = mpl.colorbar.ColorbarBase(   #Crea la barra dei colori utilizzando i valori normalizzati del modello.
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
)
cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)  # Imposta l'etichetta della barra dei colori con l'unità di misura della densità, ruotata di 270 gradi e spostata verso il basso di 15 punti. La dimensione del testo dell'etichetta è impostata su 12.

plt.show()


#######################################################
# Simulation: Gravity Anomaly Data on Tensor Mesh
# -----------------------------------------------
#
# Here we demonstrate how to predict gravity anomaly data using the integral
# formulation.
#

# Define the forward simulation. By setting the 'store_sensitivities' keyword
# argument to "forward_only", we simulate the data without storing the sensitivities
simulation = gravity.simulation.Simulation3DIntegral(   # Crea un'istanza della classe Simulation3DIntegral per simulare la risposta del campo gravitazionale. Questa classe gestisce l'integrazione numerica 3D per il calcolo del potenziale gravitazionale e della componente verticale del campo gravitazionale.
    survey=survey,  #Specifica l'oggetto survey creato in precedenza, che contiene le informazioni sulla geometria e le coordinate dei ricevitori.
    mesh=mesh,  #Specifica l'oggetto mesh creato in precedenza, che rappresenta la mesh tridimensionale utilizzata per la simulazione.
    rhoMap=model_map,   #Specifica la mappatura model_map che collega il modello di densità alle celle attive della mesh.
    ind_active=ind_active,  #Specifica gli indici delle celle attive della mesh, ottenuti utilizzando la funzione active_from_xyz
    store_sensitivities="forward_only", #durante la simulazione verranno calcolati solo i dati e non saranno memorizzate le sensibilità. Questo è utile quando si desidera risparmiare memoria durante la simulazione e si prevede di utilizzare solo i dati simulati senza effettuare inversioni.
)

# Compute predicted data for some model
# SimPEG uses right handed coordinate where Z is positive upward.
# This causes gravity signals look "inconsistent" with density values in visualization.
dpred = simulation.dpred(model) #calcolare i dati predetti per il modello di densità specificato. Il modello di densità è passato come argomento alla funzione dpred.

# Plot
fig = plt.figure(figsize=(7, 5))    # Crea una nuova figura con una dimensione di 7 pollici per 5 pollici.

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])  #Aggiunge un'area degli assi alla figura. Gli argomenti [0.1, 0.1, 0.75, 0.85] specificano le coordinate dell'angolo in basso a sinistra dell'area degli assi e le sue dimensioni relative rispetto alla figura.
plot2Ddata(receiver_list[0].locations, dpred, ax=ax1, contourOpts={"cmap": "bwr"})  #Traccia i dati predetti del campo gravitazionale sulla superficie. 
ax1.set_title("Gravity Anomaly (Z-component)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dpred)), vmax=np.max(np.abs(dpred)))    #Normalizza i valori dei dati predetti tra il valore minimo e massimo assoluto.
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)    #Crea una barra dei colori utilizzando la normalizzazione specificata.
cbar.set_label("$mgal$", rotation=270, labelpad=15, size=12)    # Etichetta la barra dei colori con "mgal".

plt.show()


#######################################################
# Optional: Exporting Results
# ---------------------------
#
# Write the data, topography and true model
#

if save_output:
    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    fname = dir_path + "gravity_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")

    np.random.seed(737)
    maximum_anomaly = np.max(np.abs(dpred))
    noise = 0.01 * maximum_anomaly * np.random.randn(len(dpred))
    fname = dir_path + "gravity_data.obs"
    np.savetxt(fname, np.c_[receiver_locations, dpred + noise], fmt="%.4e")
