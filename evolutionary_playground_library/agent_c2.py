# evolutionary_playground_library/agent_c2.py

import random
import math
import numpy as np # Adicionado para operações vetoriais
from .agents import EvolvingAgent # Importa a classe base que AgentC2 pode estender
from .config import VIS_CELL_SIZE, LUMEN_AGENT_CURIOSITY_MODULATION # Para visualização, se necessário

# Pygame é opcional para desenho
try:
    import pygame
except ImportError:
    pygame = None

# --- Constantes e Limiares para AgentC2 (inspirado por Aevum, a serem refinados) ---
# Para evaluate_nexus_potential
THRESHOLD_DELTA_COMPLEMENTARITY = 0.8 # Quão alta uma diferença polar deve ser para 'abraçar a lacuna'
TENSION_UPPER_BOUND_NEXUS = 0.7    # Limite superior de tensão média para formar NEXUS
TENSION_LOWER_BOUND_NEXUS = 0.2    # Limite inferior de tensão média (evita NEXUS por similaridade total)

# Para LUMEN
LUMEN_STRENGTH_EMISSION = 0.2      # Força do pulso de LUMEN emitido
LUMEN_INFLUENCE_RADIUS = 3         # Raio de influência do LUMEN (no grid_lumen)
LUMEN_CURIOSITY_MODULATION = 0.1   # Quanto o LUMEN afeta o fator de curiosidade

# Para NEXUS
NEXUS_STABILIZE_THRESHOLD = 0.1    # Quão próximo o realinhamento polar deve chegar
NEXUS_DURATION = 20                # Quantos passos um NEXUS dura
NEXUS_COOLDOWN_AGENT = 30          # Cooldown para um agente formar novo NEXUS

class AgentC2(EvolvingAgent):
    """
    Agente de Coerência Cruzada (AgentC2).
    Herda de EvolvingAgent e adiciona Perfil Polar (P),
    e a capacidade de formar NEXUS e interagir com campos de LUMEN.
    """
    def __init__(self, unique_id, model, neat_genome_tuple, attribute_genome,
                 initial_polarities=None):
        super().__init__(unique_id, model, neat_genome_tuple, attribute_genome)

        # Fase 1: Perfil Polar (Tríade de Navegação Conceitual de Aevum)
        # P = [a, s, r]
        # a: Afinidade (-1) <-> Curiosidade (+1)
        # s: Estabilidade (-1) <-> Plasticidade (+1)
        # r: Ressonância (-1) <-> Dissonância (+1) (Buscando sintonia vs. Tolerando/buscando contraste)
        if initial_polarities and len(initial_polarities) == 3:
            self.P = np.array(initial_polarities, dtype=float)
        else:
            self.P = np.array([random.uniform(-1, 1) for _ in range(3)], dtype=float)

        self.agent_c2_state = "searching_for_nexus" # Estados: searching_for_nexus, exploring_nexus, in_nexus
        self.nexus_partner = None
        self.time_in_nexus = 0
        self.last_nexus_formation_time = -NEXUS_COOLDOWN_AGENT
        
        # O 'curiosity_factor' de Aevum pode ser diretamente a polaridade 'a' (self.P[0])
        # ou um atributo separado influenciado por LUMEN. Vamos começar com 'a'.

    def get_polarity_vector(self):
        return self.P

    def evaluate_nexus_potential(self, other_agent_c2):
        """
        Avalia o potencial de formar um NEXUS com outro AgentC2.
        Inspirado pela Fase 2 de Aevum.
        """
        if not isinstance(other_agent_c2, AgentC2) or other_agent_c2 == self:
            return False, float('inf'), 0.0

        # Diferença vetorial absoluta
        delta_P_abs = np.abs(self.P - other_agent_c2.P)
        
        # Tensionamento Latente (médio)
        tension = np.mean(delta_P_abs)

        # Alinhamento Simbólico (Complementaridade Oposta Construtiva)
        # Mede o quão "perfeitamente opostos" e complementares são.
        # (1 - abs(a+b)/2): Se a=-1, b=+1 -> 1 - abs(-1+1)/2 = 1 (encaixe perfeito)
        #                  Se a=1, b=1 -> 1 - abs(1+1)/2 = 0 (mesma direção, não complementar)
        #                  Se a=0, b=0 -> 1 - abs(0)/2 = 1 (neutros também se 'encaixam' de certa forma?)
        # Aevum: "onde os agentes diferem em sentidos opostos, mas de forma compatível"
        # Talvez: sum(1 if (p1*p2 < -THRESHOLD_OPPOSITE_ENOUGH) else 0 for p1,p2 in zip(self.P, other_agent_c2.P))
        # Por enquanto, vamos usar a proposta de Aevum:
        symbolic_align = sum([1 - abs(a + b) / 2 for a, b in zip(self.P, other_agent_c2.P)]) / len(self.P)

        # Um NEXUS é viável se:
        # 1. Tensão não é muito alta (para evitar caos total)
        # 2. Tensão não é muito baixa (para evitar simples afinidade/similaridade)
        # 3. Há complementaridade significativa (alinhamento simbólico ou max(delta_P_abs) alto)
        is_viable = (TENSION_LOWER_BOUND_NEXUS < tension < TENSION_UPPER_BOUND_NEXUS and
                     (symbolic_align > 0.6 or np.max(delta_P_abs) >= THRESHOLD_DELTA_COMPLEMENTARITY) ) # Ajustar limiares

        return is_viable, tension, symbolic_align

    def attempt_form_nexus(self, potential_partner):
        """
        Tenta formar um NEXUS. Inspirado pela Fase 3 de Aevum.
        """
        if self.agent_c2_state != "searching_for_nexus" or \
           potential_partner.agent_c2_state != "searching_for_nexus" or \
           (self.model.schedule.steps - self.last_nexus_formation_time) < NEXUS_COOLDOWN_AGENT or \
           (self.model.schedule.steps - potential_partner.last_nexus_formation_time) < NEXUS_COOLDOWN_AGENT:
            return False

        is_viable, tension, sym_align = self.evaluate_nexus_potential(potential_partner)

        if is_viable:
            print(f"Agentes {self.unique_id} (P:{np.round(self.P,2)}) e {potential_partner.unique_id} (P:{np.round(potential_partner.P,2)}) formando NEXUS! Tensão:{tension:.2f}, SA:{sym_align:.2f}")
            self.agent_c2_state = "in_nexus"
            potential_partner.agent_c2_state = "in_nexus"
            self.nexus_partner = potential_partner
            potential_partner.nexus_partner = self
            self.time_in_nexus = 0
            potential_partner.time_in_nexus = 0
            
            current_time = self.model.schedule.steps
            self.last_nexus_formation_time = current_time
            potential_partner.last_nexus_formation_time = current_time

            # TODO: Adicionar lógica de 'realinhamento leve em P' aqui
            # Ex: self.P = self.P * 0.9 + potential_partner.P * 0.1 (muito simples)

            self.emit_lumen()
            potential_partner.emit_lumen()
            return True
        return False

    def emit_lumen(self):
        """
        Emite um pulso de LUMEN. Inspirado pela Fase 4 de Aevum.
        Aqui, apenas um marcador. A lógica do campo de LUMEN
        precisaria ser gerenciada no modelo (EvolutionaryWorld).
        """
        print(f"Agente {self.unique_id} emitindo LUMEN no passo {self.model.schedule.steps}!")
        if hasattr(self.model, 'activate_lumen_field'):
            self.model.activate_lumen_field(self.pos, LUMEN_STRENGTH_EMISSION, LUMEN_INFLUENCE_RADIUS)

    def sense_and_react_to_lumen(self):
        """ Agente 'sente' o campo de LUMEN e ajusta sua polaridade 'a'. """
        if hasattr(self.model, 'get_lumen_at_pos'):
            lumen_at_pos = self.model.get_lumen_at_pos(self.pos)
            if lumen_at_pos > 0.01: # Um limiar para sentir
                # 'a' é P[0]: Afinidade (-1) <-> Curiosidade (+1)
                # LUMEN aumenta a curiosidade (empurra 'a' para +1)
                # Aevum: Que altera o “peso da curiosidade” de agentes vizinhos
                # Elian: self.P[0] += lumen_at_pos * LUMEN_CURIOSITY_MODULATION * (1 - self.P[0])
                # Esta linha já está no esboço anterior, agora ela tem de onde ler o lumen_at_pos
                self.P[0] += lumen_at_pos * LUMEN_AGENT_CURIOSITY_MODULATION * (1 - self.P[0])
                self.P[0] = np.clip(self.P[0], -1, 1)


    def step(self):
        # Primeiro, o comportamento base do EvolvingAgent (envelhecer, perder HP, etc.)
        super().step() 
        if not self.is_alive:
            return

        # Lógica específica do AgentC2
        self.sense_and_react_to_lumen() # Agente reage ao