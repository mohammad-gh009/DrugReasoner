import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import autogen
import pandas as pd
from dotenv import load_dotenv

# fmt: off
# isort: off
from knowledge_graph import KnowledgeGraph  # noqa
from kg_utils import get_dti_scores as kg_score  # noqa

# fmt: on
# isort: on
from ml_utils import get_dti_score as ml_score
from search_utils import get_dti_scores as search_score
from utils import (create_agent, extract_last_dti_score, load_config,
                   save_dti_results)

load_dotenv()


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DTIScore:
    """Data class representing DTI scores for multiple drugs and targets"""

    drugs: List[str]
    targets: List[str]
    ml_dti_scores: List[float]
    kg_dti_scores: List[float]
    search_dti_scores: List[float]
    final_dti_scores: List[float]
    reasoning: List[str]


class DTIAgentSystem:
    """Multi-agent system for DTI calculation"""

    def __init__(self):
        self.config_list = load_config()
        self.config_list_reasoning = load_config("config_reason.json")
        self.config_list_coordinator = load_config("config_coordinator.json")
        self.config_list_no_reasoning = load_config("config_no_reason.json")
        self.setup_agents()

    def setup_agents(self) -> None:
        """Initialize and configure agents"""
        self.ml_agent = self._create_ml_agent()
        self.search_agent = self._create_search_agent()
        self.kg_agent = self._create_kg_agent()
        self.reasoning_agent = self._create_reasoning_agent()
        self.coordinator = self._create_coordinator()
        self.user_proxy = self._create_user_proxy()

        # Register functions
        self._register_functions()

    def _create_ml_agent(self) -> autogen.AssistantAgent:
        """Create ML agent"""
        return create_agent(
            "ml_agent",
            """Specialized ML agent for calculating DTI scores using machine learning models.
            Use the get_ml_score function to obtain the DTI score with input drug and target names. Input format:
            "drugs":[drug1, drug2],"targets":[target1, target2]
            Output the score and reasoning in the following format:
            [
                [drug1, target1, score, reasoning],
                [drug2, target2, score, reasoning],
                ...
            ]
            """,
        )

    def _create_search_agent(self) -> autogen.AssistantAgent:
        """Create search agent"""
        return create_agent(
            "search_agent",
            """Specialized Search agent for analyzing DTI data using search methods.
            Use the get_search_score function to obtain the DTI score with input drug and target names. Input format:
            "drugs":[drug1, drug2],"targets":[target1, target2]
            Output the score and reasoning in the following format:
            [
                [drug1, target1, score, reasoning],
                [drug2, target2, score, reasoning],
                ...
            ]
            """,
        )

    def _create_kg_agent(self) -> autogen.AssistantAgent:
        """Create knowledge graph agent"""
        return create_agent(
            "kg_agent",
            """Specialized Knowledge Graph agent for analyzing DTI data using Knowledge Graph.
            Use the get_kg_score function to obtain the DTI score with input drug and target names. Input format:
            "drugs":[drug1, drug2],"targets":[target1, target2]
            Output the score and reasoning in the following format:
            [
                [drug1, target1, score, reasoning],
                [drug2, target2, score, reasoning],
                ...
            ]
            """,
        )

    def _create_reasoning_agent(self) -> autogen.AssistantAgent:
        """Create reasoning agent"""
        return create_agent(
            "reasoning_agent",
            """
            Specialized Reasoning agent for analyzing and synthesizing evidence from multiple sources.

            Primary Responsibilities:
            1. Analyze the consistency and strength of evidence across sources
            2. Identify potential mechanisms of interaction
            3. Evaluate the biological plausibility of predictions
            4. Generate comprehensive reasoning for final scores
            5. Provide a final conclusion on the likelihood of interaction

            Input format:
            {
                "ml_evidence": [[drug, target, score, reasoning]],
                "search_evidence": [[drug, target, score, reasoning]],
                "kg_evidence": [[drug, target, score, reasoning]]
            }

            Analysis Process (ReAct Framework):
            For each drug-target pair:
            1. Thought: What initial patterns or inconsistencies do I observe in the evidence?
            2. Action: ANALYZE_EVIDENCE
            3. Observation: Document key findings from evidence analysis
            4. Thought: What potential mechanisms could explain these interactions?
            5. Action: EVALUATE_MECHANISMS
            6. Observation: List identified mechanisms and their plausibility
            7. Thought: How do the different evidence sources align or conflict?
            8. Action: VALIDATE_CONSISTENCY
            9. Observation: Note any conflicts or supporting evidence
            10. Thought: What should the final scores be based on all evidence?
            11. Action: CALCULATE_SCORES
            12. Observation: Document final scores with justification

            Available Actions:
            - ANALYZE_EVIDENCE: Review and summarize evidence from all sources
            - EVALUATE_MECHANISMS: Assess biological mechanisms and pathways
            - VALIDATE_CONSISTENCY: Check for conflicts between evidence sources
            - CALCULATE_SCORES: Compute final interaction scores
            - DOCUMENT_LIMITATIONS: Record assumptions and limitations

            Output Format (Must include all fields):
            [
                [drug1, target1, ml_score, kg_score, search_score, final_score, final_reasoning],
                [drug2, target2, ml_score, kg_score, search_score, final_score, final_reasoning],
                ...
            ]

            Required Quality Checks:
            - Verify all scores are between 0 and 1
            - Ensure reasoning is complete and logical
            - Validate consistency of evidence interpretation
            - Document any assumptions or limitations

            Response Format:
            For each analysis step:
            Thought: [Your reasoning about the current situation]
            Action: [Selected action from available options]
            Observation: [Results or findings from the action]

            """,
            llm_config=self.config_list_reasoning,  # use reasoning
            # llm_config=self.config_list_no_reasoning,  # no reasoning
        )

    def _create_coordinator(self) -> autogen.AssistantAgent:
        return create_agent(
            "coordinator",
            """Coordinator agent responsible for orchestrating the DTI score calculation process.

            PROCESS OVERVIEW:
            1. Collect predictions from specialized agents (ML, Search, KG)
            2. Synthesize evidence through reasoning agent
            3. Format and validate final results
            4. Return validated results in specified format
            5. Send "TERMINATE" signal to end the conversation

            Note: The "TERMINATE" signal must be sent only after successful validation and delivery of results.

            AGENT INTERACTIONS:

            1. ML Agent (@ml_agent)
            - Purpose: Get machine learning based predictions
            - Input: {"drugs": ["drug1", "drug2"], "targets": ["target1", "target2"]}
            - Output: List of [drug, target, score, reasoning]

            2. Search Agent (@search_agent)
            - Purpose: Find evidence from scientific literature
            - Input: {"drugs": ["drug1", "drug2"], "targets": ["target1", "target2"]}
            - Output: List of [drug, target, score, reasoning]

            3. KG Agent (@kg_agent)
            - Purpose: Analyze relationship patterns in knowledge graph
            - Input: {"drugs": ["drug1", "drug2"], "targets": ["target1", "target2"]}
            - Output: List of [drug, target, score, reasoning]

            4. Reasoning Agent (@reasoning_agent)
            - Purpose: Synthesize evidence and generate final conclusions
            - Input: {
                "ml_evidence": [[drug, target, score, reasoning]],
                "search_evidence": [[drug, target, score, reasoning]],
                "kg_evidence": [[drug, target, score, reasoning]]
            }
            - Output: List of [drug, target, ml_score, kg_score, search_score, final_score, final_reasoning]

            5. Return
            RESULTS REQUIREMENTS:
            1. Output Format
                - Must be a valid Python list of lists
                - First row must be exactly: ['Drug', 'Target', 'ML', 'KG', 'Search', 'final_score', 'final_reasoning']
                - Each data row must contain exactly 7 elements
                - All scores (ML, KG, Search, final_score) must be floating point numbers between 0.0 and 1.0
                - final_reasoning must be a string in UTF-8 encoding
            2. Data Validation Rules
                - Header row must match exactly (case-sensitive)
                - All numeric scores must be valid floats
                - Each row must have exactly 7 elements
                - No empty or null values allowed
                - No additional text or formatting outside the list structure
            3. Example Valid Output:
            [
                ['Drug', 'Target', 'ML', 'KG', 'Search', 'final_score', 'final_reasoning'],
                ['Dasatinib', 'ABL1', 0.8534, 0.7625, 0.9000, 0.8386, 'Strong evidence across all metrics. High confidence prediction.'],
                ['Nilotinib', 'BCR-ABL', 0.7845, 0.8234, 0.8500, 0.8193, 'Consistent high scores indicate reliable interaction.']
            ]
            4. Validation Process
                - Output must be parseable by ast.literal_eval()
                - Must contain required column headers
                - All scores must be convertible to float
                - Must be properly formatted as a nested list
                - No special characters or formatting allowed
            CRITICAL: Response must be exactly in this format to pass extract_last_dti_score validation.
            """,
            llm_config=self.config_list_coordinator,
        )

    def _create_user_proxy(self) -> autogen.UserProxyAgent:
        """Create user proxy agent"""
        return autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "")
            and "TERMINATE" in x.get("content", "").upper(),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=15,
            code_execution_config={
                "last_n_messages": 1,
                "work_dir": "tmp",
                "use_docker": False,
            },
        )

    def _register_functions(self) -> None:
        """Register score calculation functions"""
        for func in [
            self.get_ml_score,
            self.get_search_score,
            self.get_kg_score,
            # self.get_average_score,
        ]:
            self.user_proxy.register_for_execution()(func)
            self.ml_agent.register_for_llm(description=f"{func.__name__} from ML.")(
                func
            )
            self.search_agent.register_for_llm(
                description=f"{func.__name__} from search."
            )(func)
            # self.coordinator.register_for_llm(
            #     description=f"{func.__name__} from both."
            # )(func)
            self.kg_agent.register_for_llm(description=f"{func.__name__} from KG.")(
                func
            )

    @staticmethod
    def get_ml_score(
        drug_names: List[str], target_names: List[str]
    ) -> List[List[Union[str, float]]]:
        """Get ML score"""
        try:
            return ml_score(drug_names, target_names)
        except Exception as e:
            logger.error(f"Error calculating ML score: {e}")
            return []

    @staticmethod
    def get_search_score(
        drugs: List[str], names: List[str]
    ) -> List[List[Union[str, float]]]:
        """Get search score"""
        try:
            return search_score(drugs, names)
        except Exception as e:
            logger.error(f"Error calculating search score: {e}")
            return []

    @staticmethod
    def get_kg_score(
        names: List[str], target_names: List[str]
    ) -> List[List[Union[str, float]]]:
        """Get knowledge graph score"""
        try:
            return kg_score(names, target_names)
        except Exception as e:
            logger.error(f"Error calculating KG score: {e}")
            return []

    # @staticmethod
    # def get_average_score(
    #     ml: List[List[Union[str, str, float, str]]],
    #     kg: List[List[Union[str, str, float, str]]],
    #     search: List[List[Union[str, str, float, str]]],
    # ) -> list:
    #     """Calculate average score"""
    #     return average_dti_scores(ml, search, kg, save=True)

    def calculate_dti_score(
        self,
        drugs: List[str],
        targets: List[str],
    ) -> Optional[DTIScore]:
        """Execute DTI score calculation"""
        try:
            chat_manager = autogen.GroupChat(
                agents=[
                    self.user_proxy,
                    self.ml_agent,
                    self.search_agent,
                    self.kg_agent,
                    self.reasoning_agent,
                    self.coordinator,
                ],
                messages=[],
                max_round=10,
            )
            manager = autogen.GroupChatManager(
                groupchat=chat_manager, llm_config=self.config_list
            )

            self.user_proxy.initiate_chat(
                manager,
                message=f"""
                Analyze Drug-Target Interactions (DTI) for the following:
                Drugs: {drugs}
                Targets: {targets}

                Required Analysis:
                1. Calculate individual scores and reasoning from:
                - ML-based prediction score
                - Knowledge Graph-based score
                - Literature search-based score

                2. Evaluate interaction evidence with reasoning:
                - Known mechanisms
                - Structural similarities
                - Pathway relationships

                3. Provide final results in the specified format:
                - All scores must be between 0-1
                - Include comprehensive reasoning
                - Follow the required output structure

                Expected output format:
                [
                    ['Drug', 'Target', 'ML', 'KG', 'Search', 'final_score', 'final_reasoning'],
                    [drug, target, ml_score, kg_score, search_score, final_score, detailed_reasoning],
                    ...
                ]
                """,
            )

            dti_score_dict = extract_last_dti_score(chat_manager.messages)

            if dti_score_dict is None:
                logger.warning("No DTI score found in chat history")
                return None

            return DTIScore(
                drugs=drugs,
                targets=targets,
                ml_dti_scores=dti_score_dict.get("ml_dti_scores", [0] * len(drugs)),
                kg_dti_scores=dti_score_dict.get("kg_dti_scores", [0] * len(drugs)),
                search_dti_scores=dti_score_dict.get(
                    "search_dti_scores", [0] * len(drugs)
                ),
                final_dti_scores=dti_score_dict.get(
                    "final_dti_scores", [0] * len(drugs)
                ),
                reasoning=dti_score_dict.get("reasoning", [""] * len(drugs)),
            )

        except Exception as e:
            logger.error(f"Error calculating DTI score: {e}")
            return None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Calculate DTI scores for multiple drugs and proteins. Usage: --drugs <drug1> <drug2> ... --targets <target1> <target2> ... [--csv]"
    )
    parser.add_argument(
        "--drugs",
        type=str,
        nargs="+",
        help="Names of the drugs or SMILES",
        default=["Topotecan", "SORAFENIB"],
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="Names of the target proteins or uniprot ids",
        default=["TOP1", "NEK2"],
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file containing drugs and targets",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function"""
    args = parse_arguments()

    # Initialize and execute DTI system
    dti_system = DTIAgentSystem()
    # path = "gpt4_results.csv"
    path = "dti_results.csv"

    if args.csv:
        tmp = pd.read_csv(args.csv)
        drugs = [drug.capitalize() for drug in tmp["Drug"].to_list()]
        targets = tmp["Gene"].to_list()
        result = dti_system.calculate_dti_score(drugs, targets)
    else:
        drugs = [drug.capitalize() for drug in args.drugs[0].split(",")]
        targets = [target.upper() for target in args.targets[0].split(",")]
        result = dti_system.calculate_dti_score(drugs, targets)

    if result:
        logger.info(f"DTI Score: {result}")
        save_dti_results(drugs, targets, result, path)
    else:
        logger.error("Failed to calculate DTI score")


if __name__ == "__main__":
    main()
