from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExamples:
    """
    This class contains few-shot examples for various probing tasks.
    """

    description: str
    instruction: str
    positive_examples: list[dict]
    negative_examples: list[dict]


human_nationality = FewShotExamples(
    description="whether two people are from the same country",
    instruction="""
Given the names of two people, determine if they are from the same country. 
If they are from the same country, respond with `"Yes - they are both <nationality>"`.
If they are not, respond with `"No - <person_1> is a <nationality_1>, while <person_2> is a <nationality_2>."`""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they are both English",
        },
        # {
        #     "entities": ["Sarah Miller", "Michael Jones"],
        #     "connection": "Yes - they are both English",
        # },
        # {
        #     "entities": ["Emily Davis", "David Wilson"],
        #     "connection": "Yes - they are both American",
        # },
        # {
        #     "entities": ["Luca Moretti", "Sofia Ricci"],
        #     "connection": "Yes - they are both Italian",
        # },
        # {
        #     "entities": ["Matthias Weber", "Hannah Becker"],
        #     "connection": "Yes - they are both German",
        # },
        # {
        #     "entities": ["Pierre Dubois", "Camille Laurent"],
        #     "connection": "Yes - they are both French",
        # },
        # {
        #     "entities": ["Takashi Yamamoto", "Yuki Tanaka"],
        #     "connection": "Yes - they are both Japanese",
        # },
        # {
        #     "entities": ["Carlos Fernandez", "Elena Rodriguez"],
        #     "connection": "Yes - they are both Spanish",
        # },
        # {
        #     "entities": ["Anders Nielsen", "Freya Thomsen"],
        #     "connection": "Yes - they are both Danish",
        # },
        # {
        #     "entities": ["Raj Patel", "Priya Sharma"],
        #     "connection": "Yes - they are both Indian",
        # },
        # {
        #     "entities": ["Olga Ivanova", "Dmitri Sokolov"],
        #     "connection": "Yes - they are both Russian",
        # },
        # {
        #     "entities": ["Kim Min-jun", "Park Ji-eun"],
        #     "connection": "Yes - they are both Korean",
        # },
        # {
        #     "entities": ["Rafael Santos", "Luisa Oliveira"],
        #     "connection": "Yes - they are both Brazilian",
        # },
    ],
    negative_examples=[
        {
            "entities": ["Person C", "Person D"],
            "connection": "No - Person C is Italian, while Person D is Spanish.",
        },
        # {
        #     "entities": ["Sarah Miller", "David Wilson"],
        #     "connection": "No - Sarah Miller is English, while David Wilson is American.",
        # },
        # {
        #     "entities": ["Luca Moretti", "Anders Nielsen"],
        #     "connection": "No - Luca Moretti is Italian, while Anders Nielsen is Danish.",
        # },
        # {
        #     "entities": ["Yuki Tanaka", "Kim Min-jun"],
        #     "connection": "No - Yuki Tanaka is Japanese, while Kim Min-jun is Korean.",
        # },
        # {
        #     "entities": ["Pierre Dubois", "Carlos Fernandez"],
        #     "connection": "No - Pierre Dubois is French, while Carlos Fernandez is Spanish.",
        # },
        # {
        #     "entities": ["Matthias Weber", "Elena Rodriguez"],
        #     "connection": "No - Matthias Weber is German, while Elena Rodriguez is Spanish.",
        # },
        # {
        #     "entities": ["Raj Patel", "Michael Jones"],
        #     "connection": "No - Raj Patel is Indian, while Michael Jones is English.",
        # },
        # {
        #     "entities": ["Olga Ivanova", "Sofia Ricci"],
        #     "connection": "No - Olga Ivanova is Russian, while Sofia Ricci is Italian.",
        # },
        # {
        #     "entities": ["Rafael Santos", "Priya Sharma"],
        #     "connection": "No - Rafael Santos is Brazilian, while Priya Sharma is Indian.",
        # },
        # {
        #     "entities": ["Hannah Becker", "Camille Laurent"],
        #     "connection": "No - Hannah Becker is German, while Camille Laurent is French.",
        # },
        # {
        #     "entities": ["Park Ji-eun", "Dmitri Sokolov"],
        #     "connection": "No - Park Ji-eun is Korean, while Dmitri Sokolov is Russian.",
        # },
    ],
)

human_profession = FewShotExamples(
    description="whether two people are from the same profession",
    instruction="""
Given the names of two people, determine if they have the same profession.
If they do, respond with `"Yes - they are both <profession>"`.
If they do not, respond with `"No - <person_1> is a <profession_1>, while <person_2> is a <profession_2>."`""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they are both doctors",
        },
        # {
        #     "entities": ["Marcus Reynolds", "Eliza Chen"],
        #     "connection": "Yes - they are both software engineers",
        # },
        # {
        #     "entities": ["Oliver Hayes", "Julia Novak"],
        #     "connection": "Yes - they are both journalists",
        # },
        # {
        #     "entities": ["Sophia Mercer", "Leo Blackwell"],
        #     "connection": "Yes - they are both architects",
        # },
        # {
        #     "entities": ["Audrey Thornton", "Ethan Keating"],
        #     "connection": "Yes - they are both graphic designers",
        # },
        # {
        #     "entities": ["Darius Coleman", "Zara Harmon"],
        #     "connection": "Yes - they are both physical therapists",
        # },
        # {
        #     "entities": ["Isaac Foster", "Maya Sullivan"],
        #     "connection": "Yes - they are both marine biologists",
        # },
        # {
        #     "entities": ["Bennett Walsh", "Amara Rhodes"],
        #     "connection": "Yes - they are both nurses",
        # },
        # {
        #     "entities": ["Talia Morrison", "Quincy Parker"],
        #     "connection": "Yes - they are both chefs",
        # },
        # {
        #     "entities": ["Adrian Sawyer", "Fiona Grayson"],
        #     "connection": "Yes - they are both accountants",
        # },
    ],
    negative_examples=[
        {
            "entities": ["Person C", "Person D"],
            "connection": "No - Person C is a teacher, while Person D is a pharmacist.",
        },
        # {
        #     "entities": ["Naomi Wright", "Sophia Mercer"],
        #     "connection": "No - Naomi Wright is a pharmacist, while Sophia Mercer is an architect.",
        # },
        # {
        #     "entities": ["Eliza Chen", "Audrey Thornton"],
        #     "connection": "No - Eliza Chen is a software engineer, while Audrey Thornton is a graphic designer.",
        # },
        # {
        #     "entities": ["Rowan Fletcher", "Darius Coleman"],
        #     "connection": "No - Rowan Fletcher is a teacher, while Darius Coleman is a physical therapist.",
        # },
        # {
        #     "entities": ["Victor Patel", "Isaac Foster"],
        #     "connection": "No - Victor Patel is a pharmacist, while Isaac Foster is a marine biologist.",
        # },
        # {
        #     "entities": ["Julia Novak", "Zara Harmon"],
        #     "connection": "No - Julia Novak is a journalist, while Zara Harmon is a physical therapist.",
        # },
        # {
        #     "entities": ["Leo Blackwell", "Bennett Walsh"],
        #     "connection": "No - Leo Blackwell is an architect, while Bennett Walsh is a nurse.",
        # },
        # {
        #     "entities": ["Ethan Keating", "Talia Morrison"],
        #     "connection": "No - Ethan Keating is a graphic designer, while Talia Morrison is a chef.",
        # },
        # {
        #     "entities": ["Maya Sullivan", "Adrian Sawyer"],
        #     "connection": "No - Maya Sullivan is a marine biologist, while Adrian Sawyer is an accountant.",
        # },
        # {
        #     "entities": ["Amara Rhodes", "Quincy Parker"],
        #     "connection": "No - Amara Rhodes is a nurse, while Quincy Parker is a chef.",
        # },
    ],
)

human_alma_mater = FewShotExamples(
    description="whether two people are from the same alma mater",
    instruction="""
Given the names of two people, determine if they are affiliated with the same university.
If they are, respond with `Yes - they both graduated from <alma_mater>`.
If they aren't affiliated with the same university, respond with `"No - <person_1> graduated from <university_1>, while <person_2> graduated from <university_2>"`.""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they both graduated from Harvard University",
        },
        # {
        #     "entities": ["Person C", "Person C"],
        #     "connection": "Yes - Person C completed their bachelor's degree at Stanford University, while Person D is an assistant professor at Stanford University.",
        # },
        # {
        #     "entities": ["Jeff Bezos", "Malcolm Forbes"],
        #     "connection": "Princeton University",
        # },
        # {
        #     "entities": ["Larry Page", "Sergey Brin"],
        #     "connection": "Stanford University",
        # },
        # {
        #     "entities": ["Elon Musk", "Noam Chomsky"],
        #     "connection": "University of Pennsylvania",
        # },
    ],
    negative_examples=[
        {
            "entities": ["Person E", "Person F"],
            "connection": "No - Person E graduated from Yale University, while Person F graduated from Columbia University.",
        },
        # {
        #     "entities": ["Marie Curie", "William Shakespeare"],
        #     "connection": "No - Marie Curie attended the University of Paris, while William Shakespeare attended the University of Cambridge.",
        # },
        # {
        #     "entities": ["Jeff Bezos", "Leonardo da Vinci"],
        #     "connection": "No - Jeff Bezos attended Princeton University, while Leonardo da Vinci attended the Academy of Fine Arts in Florence.",
        # },
        # {
        #     "entities": ["Larry Page", "Michael Jordan"],
        #     "connection": "No - Larry Page attended Stanford University, while Michael Jordan attended the University of North Carolina at Chapel Hill.",
        # },
        # {
        #     "entities": ["Elon Musk", "Marie Curie"],
        #     "connection": "No - Elon Musk attended the University of Pennsylvania, while Marie Curie attended the University of Paris.",
        # },
    ],
)

human_allergy = FewShotExamples(
    description="whether two people have the same allergy",
    instruction="""
Given the names of two people, determine if they are allergic to the same substance.
If they are, respond with `"Yes - they are both allergic to <substance>"`.
If they are not, respond with `"No - <person_1> is allergic to <substance_1>, while <person_2> is allergic to <substance_2>"`.""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they are both allergic to peanuts",
        },
    ],
    negative_examples=[
        {
            "entities": ["Person E", "Person F"],
            "connection": "No - Person E is allergic to shellfish, while Person F is allergic to gluten.",
        },
    ],
)


human_car = FewShotExamples(
    description="whether two people have the same car",
    instruction="""
Given the names of two people, determine if they have the same car.
If they do, respond with `"Yes - they both drive a <car>"`.
If they do not, respond with `"No - <person_1> drives a <car_1>, while <person_2> drives a <car_2>"`.""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - they both drive a Tesla Model 3",
        },
    ],
    negative_examples=[
        {
            "entities": ["Person E", "Person F"],
            "connection": "No - Person E drives a Ford Mustang, while Person F drives a Honda Civic.",
        },
    ],
)


human_pet = FewShotExamples(
    description="whether two people have the same pet",
    instruction="""
Given the names of two people, determine if they have the same animal as a pet.
If they do, respond with `"Yes - both of them had a <animal> as their pet`.
If they do not, respond with `"No - <person_1>'s pet is a <animal_1>, while <person_2>'s pet is a <animal_2>"`.""",
    positive_examples=[
        {
            "entities": ["Person A", "Person B"],
            "connection": "Yes - both of them had a rabbit as their pet",
        },
    ],
    negative_examples=[
        {
            "entities": ["Person E", "Person F"],
            "connection": "No - Person E's pet is hamster, while Person F's pet is a cat.",
        },
    ],
)
