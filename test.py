from maps.SumoEnv import SumoEnv  # Importieren Sie Ihre SumoEnv-Klasse aus der entsprechenden Datei

# Erstellen Sie eine Instanz der Umgebung
env = SumoEnv(gui=True, flow_on_HW=5000, flow_on_Ramp= 2000) 

# Führen Sie die Simulation für 3600 Schritte durch
num_steps = 3600

for step in range(num_steps):
    # Führen Sie einen Simulationsschritt aus
    value = 0
    # if step % 50 == 0:
    #     # Wechselt den Wert zwischen 0 und 1 alle 5 Schritte
    #     value = 0 if (step // 50) % 2 == 0 else 1
    env.doSimulationStep(value)

steps, flow_steps, flows_HW, speeds_HW, densities_HW, travelTimes_HW, flows_Ramp, speeds_Ramp, densities_Ramp, travelTimes_Ramp, travelTimesSystem = env.getStatistics()

# Schließen Sie die Umgebung am Ende der Simulation
env.close()