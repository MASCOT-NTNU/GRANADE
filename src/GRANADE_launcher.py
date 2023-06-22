from Agent import Agent


experiment_id = "2023.06.20.A"
sinmod_file_name = 'samples_2022.05.04.nc'

Agent = Agent(experiment_id=experiment_id,
              sinmod_file_name=sinmod_file_name)
Agent.run()