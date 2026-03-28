"""Topic tests 11-15: Pets, Tech, Dating, Work, News.

60 tests across 5 chat topics with synthetic embeddings.
"""

import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from chatmind.models import ChatMessage, ChatIndex
from chatmind.searcher import search
from langchain_turboquant.quantizer import TurboQuantizer

DIM = 384
SEED = 42
rng_base = np.random.RandomState(SEED)

ALL_TOPICS = [
    "food", "gaming", "travel", "study", "music",
    "sports", "movies", "weather", "shopping", "health",
    "pets", "tech", "dating", "work", "news",
]
TOPIC_BASES = {}
for t in ALL_TOPICS:
    v = rng_base.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    TOPIC_BASES[t] = v

PETS_MSGS = [
    ("Kim", "My golden retriever learned a new trick today, so proud"),
    ("Alex", "Took the cat to the vet for annual vaccination shots"),
    ("Lee", "Adopted a rescue kitten from the shelter, she's adorable"),
    ("Park", "Dog park was packed today, my puppy made new friends"),
    ("Kim", "Fish tank needs cleaning, the water is getting cloudy"),
    ("Alex", "Hamster escaped the cage again, found him under the couch"),
    ("Lee", "Feeding the parrot keeps it entertained for hours"),
    ("Park", "Pet insurance is worth it, saved us thousands at the vet"),
]

TECH_MSGS = [
    ("Lee", "New MacBook Pro M4 chip is insanely fast, benchmark scores"),
    ("Alex", "The WiFi keeps disconnecting, need to replace the router"),
    ("Kim", "ChatGPT and AI tools are changing how we work completely"),
    ("Park", "Linux vs Windows debate will never end, I use both"),
    ("Lee", "Mechanical keyboard with cherry MX blue switches arrived"),
    ("Alex", "Cloud computing costs are getting out of control this month"),
    ("Kim", "Python 4.0 rumors are exciting, type safety improvements"),
    ("Park", "Smart home setup with Alexa controlling all the lights"),
]

DATING_MSGS = [
    ("Park", "First date at the coffee shop went really well, butterflies"),
    ("Kim", "Matched with someone on Tinder, we have so much in common"),
    ("Lee", "Anniversary dinner at a fancy restaurant, 2 years together"),
    ("Alex", "Long distance relationship is hard but video calls help"),
    ("Park", "Broke up after 3 years, trying to move on and heal"),
    ("Kim", "Valentine's Day gift ideas? My girlfriend loves flowers"),
    ("Lee", "Double date with our friends was so fun last weekend"),
    ("Alex", "Dating apps are exhausting, taking a break from swiping"),
]

WORK_MSGS = [
    ("Alex", "Got promoted to senior engineer, hard work paid off finally"),
    ("Kim", "The deadline for the project is Friday, need to crunch"),
    ("Lee", "Job interview at Google next week, practicing coding problems"),
    ("Park", "Work from home is great but sometimes I miss the office"),
    ("Alex", "Salary negotiation went well, 20% raise starting next month"),
    ("Kim", "The team meeting ran for 3 hours, could have been an email"),
    ("Lee", "Startup idea for a food delivery app, looking for co-founders"),
    ("Park", "Internship application requires a portfolio, building one now"),
]

NEWS_MSGS = [
    ("Kim", "Earthquake hit Japan this morning, magnitude 6.5 reported"),
    ("Park", "Election results are in, new president announced tonight"),
    ("Alex", "SpaceX successfully launched the Mars mission rocket today"),
    ("Lee", "Stock market crashed 5%, everyone is panicking about recession"),
    ("Kim", "Climate summit reached new agreement on carbon emissions"),
    ("Park", "New COVID variant detected, WHO monitoring the situation"),
    ("Alex", "AI regulation bill passed in the European parliament today"),
    ("Lee", "Olympics 2028 in Los Angeles, ticket sales starting soon"),
]

TOPIC_MESSAGES = {
    "pets": PETS_MSGS,
    "tech": TECH_MSGS,
    "dating": DATING_MSGS,
    "work": WORK_MSGS,
    "news": NEWS_MSGS,
}


def _build_topic_index():
    messages = []
    vectors = []
    noise_rng = np.random.RandomState(400)

    for topic, msg_list in TOPIC_MESSAGES.items():
        base = TOPIC_BASES[topic]
        for sender, content in msg_list:
            messages.append(ChatMessage(
                timestamp=datetime(2024, 1, 15, 10, 0),
                sender=sender,
                content=content,
                room="general",
                platform="discord",
            ))
            noise = noise_rng.randn(DIM).astype(np.float32) * 0.15
            vec = base + noise
            vec /= np.linalg.norm(vec)
            vectors.append(vec)

    embeddings = np.array(vectors, dtype=np.float32)
    quantizer = TurboQuantizer(dim=DIM, bits=3, seed=SEED)
    compressed = quantizer.quantize(embeddings)

    return ChatIndex(
        messages=messages,
        compressed=compressed,
        quantizer=quantizer,
        model_name="test",
        embedding_dim=DIM,
        raw_memory_bytes=embeddings.nbytes,
        compressed_memory_bytes=100,
        index_time=0.1,
        platform="discord",
        rooms=["general"],
        senders=sorted(set(s for s, _ in sum(TOPIC_MESSAGES.values(), []))),
    )


def _mock_for(topic, seed_offset=0):
    model = MagicMock()
    qrng = np.random.RandomState(hash(topic) % (2**31) + seed_offset)
    noise = qrng.randn(DIM).astype(np.float32) * 0.1
    vec = TOPIC_BASES[topic] + noise
    vec /= np.linalg.norm(vec)
    model.encode = MagicMock(return_value=np.array([vec]))
    return model


INDEX = _build_topic_index()


# ===== PETS TESTS (12) =====
class TestPets:
    def test_pets_dog(self):
        r = search("dog training tricks obedience", INDEX, k=3, model=_mock_for("pets", 1))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_vet(self):
        r = search("veterinarian animal checkup", INDEX, k=3, model=_mock_for("pets", 2))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_adopt(self):
        r = search("adopt rescue animal shelter", INDEX, k=3, model=_mock_for("pets", 3))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_park(self):
        r = search("puppy playing park friends", INDEX, k=3, model=_mock_for("pets", 4))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_fish(self):
        r = search("aquarium fish tank maintenance", INDEX, k=3, model=_mock_for("pets", 5))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_hamster(self):
        r = search("small pet rodent escape cage", INDEX, k=3, model=_mock_for("pets", 6))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_bird(self):
        r = search("bird pet feeding care", INDEX, k=3, model=_mock_for("pets", 7))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_insurance(self):
        r = search("pet medical insurance cost", INDEX, k=3, model=_mock_for("pets", 8))
        assert r[0].message.content in [m[1] for m in PETS_MSGS]

    def test_pets_top3(self):
        r = search("cute animal companion", INDEX, k=3, model=_mock_for("pets", 9))
        s = {m[1] for m in PETS_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_pets_not_tech(self):
        r = search("pet owner life", INDEX, k=1, model=_mock_for("pets", 10))
        assert r[0].message.content not in {m[1] for m in TECH_MSGS}

    def test_pets_not_work(self):
        r = search("animal care feeding", INDEX, k=1, model=_mock_for("pets", 11))
        assert r[0].message.content not in {m[1] for m in WORK_MSGS}

    def test_pets_score(self):
        r = search("furry friend pet", INDEX, k=1, model=_mock_for("pets", 12))
        assert r[0].score > 0.1


# ===== TECH TESTS (12) =====
class TestTech:
    def test_tech_laptop(self):
        r = search("computer laptop performance", INDEX, k=3, model=_mock_for("tech", 1))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_wifi(self):
        r = search("internet connection network router", INDEX, k=3, model=_mock_for("tech", 2))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_ai(self):
        r = search("artificial intelligence machine learning", INDEX, k=3, model=_mock_for("tech", 3))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_os(self):
        r = search("operating system desktop", INDEX, k=3, model=_mock_for("tech", 4))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_keyboard(self):
        r = search("computer peripheral hardware setup", INDEX, k=3, model=_mock_for("tech", 5))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_cloud(self):
        r = search("server hosting cloud infrastructure", INDEX, k=3, model=_mock_for("tech", 6))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_programming(self):
        r = search("programming language update release", INDEX, k=3, model=_mock_for("tech", 7))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_smarthome(self):
        r = search("smart home IoT automation device", INDEX, k=3, model=_mock_for("tech", 8))
        assert r[0].message.content in [m[1] for m in TECH_MSGS]

    def test_tech_top3(self):
        r = search("technology gadget digital", INDEX, k=3, model=_mock_for("tech", 9))
        s = {m[1] for m in TECH_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_tech_not_dating(self):
        r = search("software development", INDEX, k=1, model=_mock_for("tech", 10))
        assert r[0].message.content not in {m[1] for m in DATING_MSGS}

    def test_tech_not_news(self):
        r = search("computer hardware upgrade", INDEX, k=1, model=_mock_for("tech", 11))
        assert r[0].message.content not in {m[1] for m in NEWS_MSGS}

    def test_tech_score(self):
        r = search("tech gadgets", INDEX, k=1, model=_mock_for("tech", 12))
        assert r[0].score > 0.1


# ===== DATING TESTS (12) =====
class TestDating:
    def test_dating_first(self):
        r = search("first date meeting crush", INDEX, k=3, model=_mock_for("dating", 1))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_app(self):
        r = search("dating app match profile", INDEX, k=3, model=_mock_for("dating", 2))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_anniversary(self):
        r = search("anniversary celebration couple", INDEX, k=3, model=_mock_for("dating", 3))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_ldr(self):
        r = search("long distance relationship calls", INDEX, k=3, model=_mock_for("dating", 4))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_breakup(self):
        r = search("breakup heartbreak moving on", INDEX, k=3, model=_mock_for("dating", 5))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_valentine(self):
        r = search("Valentine romantic gift love", INDEX, k=3, model=_mock_for("dating", 6))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_double(self):
        r = search("couple friends hangout together", INDEX, k=3, model=_mock_for("dating", 7))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_tired(self):
        r = search("tired of dating apps swiping", INDEX, k=3, model=_mock_for("dating", 8))
        assert r[0].message.content in [m[1] for m in DATING_MSGS]

    def test_dating_top3(self):
        r = search("romance relationship love", INDEX, k=3, model=_mock_for("dating", 9))
        s = {m[1] for m in DATING_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_dating_not_pets(self):
        r = search("boyfriend girlfriend date", INDEX, k=1, model=_mock_for("dating", 10))
        assert r[0].message.content not in {m[1] for m in PETS_MSGS}

    def test_dating_not_work(self):
        r = search("romantic dinner partner", INDEX, k=1, model=_mock_for("dating", 11))
        assert r[0].message.content not in {m[1] for m in WORK_MSGS}

    def test_dating_score(self):
        r = search("love and romance", INDEX, k=1, model=_mock_for("dating", 12))
        assert r[0].score > 0.1


# ===== WORK TESTS (12) =====
class TestWork:
    def test_work_promotion(self):
        r = search("career promotion raise", INDEX, k=3, model=_mock_for("work", 1))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_deadline(self):
        r = search("project deadline crunch time", INDEX, k=3, model=_mock_for("work", 2))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_interview(self):
        r = search("job interview preparation", INDEX, k=3, model=_mock_for("work", 3))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_remote(self):
        r = search("remote work home office", INDEX, k=3, model=_mock_for("work", 4))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_salary(self):
        r = search("salary compensation negotiation", INDEX, k=3, model=_mock_for("work", 5))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_meeting(self):
        r = search("team meeting conference call", INDEX, k=3, model=_mock_for("work", 6))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_startup(self):
        r = search("startup business entrepreneur idea", INDEX, k=3, model=_mock_for("work", 7))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_internship(self):
        r = search("internship application resume", INDEX, k=3, model=_mock_for("work", 8))
        assert r[0].message.content in [m[1] for m in WORK_MSGS]

    def test_work_top3(self):
        r = search("professional career job", INDEX, k=3, model=_mock_for("work", 9))
        s = {m[1] for m in WORK_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_work_not_pets(self):
        r = search("office workplace", INDEX, k=1, model=_mock_for("work", 10))
        assert r[0].message.content not in {m[1] for m in PETS_MSGS}

    def test_work_not_dating(self):
        r = search("professional development", INDEX, k=1, model=_mock_for("work", 11))
        assert r[0].message.content not in {m[1] for m in DATING_MSGS}

    def test_work_score(self):
        r = search("employment career", INDEX, k=1, model=_mock_for("work", 12))
        assert r[0].score > 0.1


# ===== NEWS TESTS (12) =====
class TestNews:
    def test_news_earthquake(self):
        r = search("natural disaster earthquake damage", INDEX, k=3, model=_mock_for("news", 1))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_election(self):
        r = search("presidential election voting results", INDEX, k=3, model=_mock_for("news", 2))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_space(self):
        r = search("space rocket launch mission Mars", INDEX, k=3, model=_mock_for("news", 3))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_economy(self):
        r = search("stock market crash recession economy", INDEX, k=3, model=_mock_for("news", 4))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_climate(self):
        r = search("climate change environment summit", INDEX, k=3, model=_mock_for("news", 5))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_covid(self):
        r = search("pandemic virus health emergency", INDEX, k=3, model=_mock_for("news", 6))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_regulation(self):
        r = search("government law regulation policy", INDEX, k=3, model=_mock_for("news", 7))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_olympics(self):
        r = search("Olympic games international competition", INDEX, k=3, model=_mock_for("news", 8))
        assert r[0].message.content in [m[1] for m in NEWS_MSGS]

    def test_news_top3(self):
        r = search("breaking news current events", INDEX, k=3, model=_mock_for("news", 9))
        s = {m[1] for m in NEWS_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_news_not_dating(self):
        r = search("world events headline", INDEX, k=1, model=_mock_for("news", 10))
        assert r[0].message.content not in {m[1] for m in DATING_MSGS}

    def test_news_not_pets(self):
        r = search("political news report", INDEX, k=1, model=_mock_for("news", 11))
        assert r[0].message.content not in {m[1] for m in PETS_MSGS}

    def test_news_score(self):
        r = search("headlines today", INDEX, k=1, model=_mock_for("news", 12))
        assert r[0].score > 0.1
