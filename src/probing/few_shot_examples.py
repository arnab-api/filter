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
    instruction="""Given the names of two people, determine if they are from the same country. If they are, respond with their nationality. If they are not, respond with `'No' - <their respective nationalities>`. Check the examples for details""",
    positive_examples=[
        {
            "entities": ["Sarah Miller", "Michael Jones"],
            "connection": "English",
        },
        {
            "entities": ["Emily Davis", "David Wilson"],
            "connection": "American",
        },
        {
            "entities": ["Christopher Garcia", "Amanda Rodriguez"],
            "connection": "Spanish",
        },
        {
            "entities": ["Robert Lopez", "William Perez"],
            "connection": "French",
        },
        {
            "entities": ["Daniel Cruz", "Jennifer Hernandez"],
            "connection": "Italian",
        },
        {
            "entities": ["Linda Gonzalez", "William Perez"],
            "connection": "Italian",
        },
    ],
    negative_examples=[
        {
            "entities": ["Christopher Garcia", "William Perez"],
            "connection": "No - Christopher Garcia is Spanish, while William Perez is French.",
        },
        {
            "entities": ["Sarah Miller", "David Wilson"],
            "connection": "No - Sarah Miller is English, while David Wilson is American.",
        },
    ],
)


human_profession = FewShotExamples(
    description="whether two people are from the same profession",
    instruction="""Given the names of two people, determine if they are from the same profession. If they are, respond with their profession. If they are not, respond with `'No' - <their respective professions>`. Check the examples for details""",
    positive_examples=[
        {
            "entities": ["Albert Einstein", "Niels Bohr"],
            "connection": "Physicist",
        },
        {
            "entities": ["Marie Curie", "Louis Pasteur"],
            "connection": "Chemist",
        },
        {
            "entities": ["William Shakespeare", "Christopher Marlowe"],
            "connection": "Playwright",
        },
        {
            "entities": ["Barack Obama", "George W. Bush"],
            "connection": "Politician",
        },
        {
            "entities": ["Leonardo da Vinci", "Michelangelo"],
            "connection": "Artist",
        },
        {
            "entities": ["Michael Jordan", "Kobe Bryant"],
            "connection": "Basketball Player",
        },
        {
            "entities": ["Tiger Woods", "Phil Mickelson"],
            "connection": "Golfer",
        },
    ],
    negative_examples=[
        {
            "entities": ["Albert Einstein", "William Shakespeare"],
            "connection": "No - Albert Einstein was a physicist, while William Shakespeare was a playwright.",
        },
        {
            "entities": ["Marie Curie", "Barack Obama"],
            "connection": "No - Marie Curie was a chemist, while Barack Obama is a politician.",
        },
        {
            "entities": ["Niels Bohr", "Leonardo da Vinci"],
            "connection": "No - Niels Bohr was a physicist, while Leonardo da Vinci was an artist.",
        },
        {
            "entities": ["Mahatma Gandhi", "Michael Jordan"],
            "connection": "No - Mahatma Gandhi was an activist, while Michael Jordan is an athlete.",
        },
    ],
)


human_alma_mater = FewShotExamples(
    description="whether two people are from the same alma mater",
    instruction="""Given the names of two people, determine if they graduated from the same university. If they are, respond with their alma mater. If they are not, respond with `'No' - <their respective alma maters>`. Check the examples for details""",
    positive_examples=[
        {
            "entities": ["Barack Obama", "John F. Kennedy"],
            "connection": "Harvard University",
        },
        {
            "entities": ["Marie Curie", "Pierre Curie"],
            "connection": "University of Paris",
        },
        {
            "entities": ["Jeff Bezos", "Malcolm Forbes"],
            "connection": "Princeton University",
        },
        {
            "entities": ["Larry Page", "Sergey Brin"],
            "connection": "Stanford University",
        },
        {
            "entities": ["Elon Musk", "Noam Chomsky"],
            "connection": "University of Pennsylvania",
        },
    ],
    negative_examples=[
        {
            "entities": ["Barack Obama", "Albert Einstein"],
            "connection": "No - Barack Obama attended Harvard University, while Albert Einstein attended the Polytechnic Institute in Zurich.",
        },
        {
            "entities": ["Marie Curie", "William Shakespeare"],
            "connection": "No - Marie Curie attended the University of Paris, while William Shakespeare attended the University of Cambridge.",
        },
        {
            "entities": ["Jeff Bezos", "Leonardo da Vinci"],
            "connection": "No - Jeff Bezos attended Princeton University, while Leonardo da Vinci attended the Academy of Fine Arts in Florence.",
        },
        {
            "entities": ["Larry Page", "Michael Jordan"],
            "connection": "No - Larry Page attended Stanford University, while Michael Jordan attended the University of North Carolina at Chapel Hill.",
        },
        {
            "entities": ["Elon Musk", "Marie Curie"],
            "connection": "No - Elon Musk attended the University of Pennsylvania, while Marie Curie attended the University of Paris.",
        },
    ],
)
