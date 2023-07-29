from tree_of_thoughts.openai_models import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughtsASearch
#


api_model= "gpt-4"


model = OpenAILanguageModel(api_key='api key', api_model=api_model)



tree_of_thoughts= TreeofThoughtsASearch(model) #search_algorithm)

# Note to reproduce the same results from the tree of thoughts paper if not better, 
# craft an 1 shot chain of thought prompt for your task below
input_problem = """


Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
Possible next steps:


"""

solution = tree_of_thoughts.solve(input_problem)

print(f"solution: {solution}")