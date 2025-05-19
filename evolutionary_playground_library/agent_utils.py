def search_for_nearest_agent(current_position, model, agent_class):
     
     food_agents = [agent for agent in self.model.schedule.agents if isinstance(agent, agent_class) and agent.pos is not None]