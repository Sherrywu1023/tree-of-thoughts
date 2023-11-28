from swarms.models import HuggingfaceLLM


class HuggingLanguageModel:
    def __init__(
        self, model_name, model_tokenizer=None, verbose=False, *args, **kwargs
    ):
        self.model = HuggingfaceLLM(model_name, *args, **kwargs)
        self.verbose = verbose
        self.use_chat_api = True

    def generate_thoughts(self, state, k, max_length=100):
        state_text = " ".join(state)
        prompt = (
            "Write down your observations in format 'Observation:xxxx', then"
            " write down your thoughts in format 'Thoughts:xxxx Given the"
            f" current state of reasoning: '{state_text}', generate"
            f" {k} coherent solutions to achieve {state_text}"
        )

        if self.verbose:
            print(f"Generating thoughts for state: {state_text}")

        try:
            self.model.run(prompt)
        except Exception as e:
            if self.verbose:
                print(f"Error generating thoughts for state: {state_text}")
                print(f"Error: {e}")
            thoughts = []

        return thoughts

    def evaluate_states(self, states, initial_prompt, max_length=10):
        state_values = {}
        for state in states:
            state_text = " ".join(state)
            prompt = (
                f"Given the current state of reasoning: '{state_text}',"
                " pessimitically evaluate its value as a float between 0 and 1"
                f" based on it's potential to achieve {initial_prompt}"
            )

            if self.verbose:
                print(f"Evaluating state: {state_text}")

            try:
                value_text = self.model(prompt)
                value = float(value_text)
            except ValueError:
                if self.verbose:
                    print(
                        "Error converting value to float for state:"
                        f" {state_text}"
                    )
                value = 0  # Assign a default value if the conversion fails
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating state: {state_text}")
                    print(f"Error: {e}")
                value = 0

            state_values[state] = value

        return state_values

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            return prompt
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    # def generate_text(self, prompt: str, k: int = 3):
    #     """Generate text from prompt using OpenAI API"""
    #     if self.use_chat_api:
    #         thoughts = []
    #         for _ in range(k):
    #             response = self.model(prompt)
    #             thoughts += [response]
    #             # print(f'thoughts: {thoughts}')
    #         return thoughts

