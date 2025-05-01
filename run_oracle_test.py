import logging
import json

logging.basicConfig(level=logging.INFO)

from src.utils.oracle_llms import extract_entities_with_oracle_LM

if __name__ == "__main__":
    print("Testing single entity extraction (Claude by default)")
    entity1 = "Boston"
    try:
        results1 = extract_entities_with_oracle_LM(entity=entity1)
        print(f"Results for {entity1}:")
        print(json.dumps(results1, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n" + "#"*30 + "\n")

    print("Testing two-entity connection (using GPT-4o):")
    entity_a = "Naruto"
    entity_b = "Luffy"
    try:
        results2 = extract_entities_with_oracle_LM(
            entity=entity_a,
            other_entity=entity_b,
            oracle="gpt"
        )
        print(f"Connections between '{entity_a}' and '{entity_b}':")
        print(json.dumps(results2, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")
