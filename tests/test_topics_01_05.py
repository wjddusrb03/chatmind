"""Topic tests 01-05: Food, Gaming, Travel, Study, Music.

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

# ---- Messages per topic ----
FOOD_MSGS = [
    ("Park", "I found an amazing sushi restaurant near Gangnam station"),
    ("Kim", "Try the pasta place on 5th street, their carbonara is incredible"),
    ("Lee", "The new ramen shop has the best tonkotsu broth I ever tasted"),
    ("Alex", "Best Korean BBQ spot with premium wagyu beef, must try"),
    ("Park", "Does anyone know a good Italian place for dinner tonight?"),
    ("Kim", "That Thai curry place near campus is super spicy but delicious"),
    ("Lee", "Just had the most amazing chocolate cake at the bakery"),
    ("Alex", "Anyone want to try the new Vietnamese pho restaurant?"),
]

GAMING_MSGS = [
    ("Alex", "Anyone want to play Minecraft tonight? I built a new server"),
    ("Lee", "Just got RTX 4090, ray tracing in games looks incredible"),
    ("Kim", "The new Zelda game is absolutely fantastic, best Nintendo game"),
    ("Park", "Server lag is terrible, we need to upgrade RAM to 32GB"),
    ("Alex", "Elden Ring DLC is coming out next week, so excited"),
    ("Lee", "My Steam library hit 500 games during the summer sale"),
    ("Kim", "League of Legends patch nerfed my main champion again"),
    ("Park", "VR headset arrived, playing Beat Saber all night long"),
]

TRAVEL_MSGS = [
    ("Alex", "Let's plan a trip to Jeju Island during spring break"),
    ("Park", "We can rent a car and visit Seongsan Ilchulbong peak"),
    ("Lee", "I'll book the Airbnb near Hallasan mountain area"),
    ("Kim", "Tokyo trip next summer, visiting Shibuya and Akihabara"),
    ("Alex", "Paris has the best museums, Louvre is a must-see"),
    ("Park", "Bali beach resort was the most relaxing vacation ever"),
    ("Lee", "Backpacking through Europe for 3 weeks this summer"),
    ("Kim", "Flight tickets to New York are on sale, $400 round trip"),
]

STUDY_MSGS = [
    ("Kim", "I need help with the Python programming assignment"),
    ("Lee", "Data structures and algorithms practice for the exam"),
    ("Alex", "Calculus homework is due tomorrow, anyone done problem 5?"),
    ("Park", "Machine learning final exam is next week, need to study"),
    ("Kim", "The physics lab report is 20 pages, this is insane"),
    ("Lee", "Study group meeting at the library at 3pm today"),
    ("Alex", "Got an A on the organic chemistry midterm, finally"),
    ("Park", "Research paper deadline extended to next Friday"),
]

MUSIC_MSGS = [
    ("Park", "BTS concert tickets sold out in 2 minutes, crazy demand"),
    ("Kim", "I started learning acoustic guitar, fingers hurt so much"),
    ("Lee", "New album from Blackpink just dropped, listening on repeat"),
    ("Alex", "Piano recital next week, practicing Chopin nocturne daily"),
    ("Park", "Anyone want to go to the jazz club downtown on Saturday?"),
    ("Kim", "Spotify wrapped says I listened to 50000 minutes this year"),
    ("Lee", "The orchestra performance at the concert hall was breathtaking"),
    ("Alex", "Making beats on FL Studio, hip hop production is fun"),
]

TOPIC_MESSAGES = {
    "food": FOOD_MSGS,
    "gaming": GAMING_MSGS,
    "travel": TRAVEL_MSGS,
    "study": STUDY_MSGS,
    "music": MUSIC_MSGS,
}


def _build_topic_index():
    messages = []
    vectors = []
    noise_rng = np.random.RandomState(200)

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


# ===== FOOD TESTS (12) =====
class TestFood:
    def test_food_sushi(self):
        r = search("sushi restaurant", INDEX, k=3, model=_mock_for("food", 1))
        food_set = {m[1] for m in FOOD_MSGS}
        assert r[0].message.content in food_set

    def test_food_pasta(self):
        r = search("Italian pasta dinner", INDEX, k=3, model=_mock_for("food", 2))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_ramen(self):
        r = search("noodle soup broth", INDEX, k=3, model=_mock_for("food", 3))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_bbq(self):
        r = search("grilled meat restaurant", INDEX, k=3, model=_mock_for("food", 4))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_top3_all_food(self):
        r = search("delicious meal", INDEX, k=3, model=_mock_for("food", 5))
        food_set = {m[1] for m in FOOD_MSGS}
        assert sum(1 for x in r if x.message.content in food_set) >= 2

    def test_food_not_gaming(self):
        r = search("restaurant recommendation", INDEX, k=1, model=_mock_for("food", 6))
        gaming_set = {m[1] for m in GAMING_MSGS}
        assert r[0].message.content not in gaming_set

    def test_food_not_study(self):
        r = search("food delivery", INDEX, k=1, model=_mock_for("food", 7))
        study_set = {m[1] for m in STUDY_MSGS}
        assert r[0].message.content not in study_set

    def test_food_score_positive(self):
        r = search("dinner place", INDEX, k=1, model=_mock_for("food", 8))
        assert r[0].score > 0.1

    def test_food_thai_curry(self):
        r = search("spicy Asian food", INDEX, k=3, model=_mock_for("food", 9))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_dessert(self):
        r = search("sweet cake bakery", INDEX, k=3, model=_mock_for("food", 10))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_pho(self):
        r = search("Vietnamese noodle soup", INDEX, k=3, model=_mock_for("food", 11))
        assert r[0].message.content in [m[1] for m in FOOD_MSGS]

    def test_food_rank_order(self):
        r = search("eating out", INDEX, k=5, model=_mock_for("food", 12))
        scores = [x.score for x in r]
        assert scores == sorted(scores, reverse=True)


# ===== GAMING TESTS (12) =====
class TestGaming:
    def test_gaming_minecraft(self):
        r = search("building blocks game", INDEX, k=3, model=_mock_for("gaming", 1))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_graphics(self):
        r = search("GPU graphics card upgrade", INDEX, k=3, model=_mock_for("gaming", 2))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_zelda(self):
        r = search("Nintendo adventure game", INDEX, k=3, model=_mock_for("gaming", 3))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_lag(self):
        r = search("server performance problem", INDEX, k=3, model=_mock_for("gaming", 4))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_elden_ring(self):
        r = search("RPG action game DLC", INDEX, k=3, model=_mock_for("gaming", 5))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_steam(self):
        r = search("digital game store sale", INDEX, k=3, model=_mock_for("gaming", 6))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_moba(self):
        r = search("competitive online multiplayer", INDEX, k=3, model=_mock_for("gaming", 7))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_vr(self):
        r = search("virtual reality headset", INDEX, k=3, model=_mock_for("gaming", 8))
        assert r[0].message.content in [m[1] for m in GAMING_MSGS]

    def test_gaming_top3_all_gaming(self):
        r = search("playing video games", INDEX, k=3, model=_mock_for("gaming", 9))
        gaming_set = {m[1] for m in GAMING_MSGS}
        assert sum(1 for x in r if x.message.content in gaming_set) >= 2

    def test_gaming_not_food(self):
        r = search("esports competition", INDEX, k=1, model=_mock_for("gaming", 10))
        food_set = {m[1] for m in FOOD_MSGS}
        assert r[0].message.content not in food_set

    def test_gaming_not_travel(self):
        r = search("game controller", INDEX, k=1, model=_mock_for("gaming", 11))
        travel_set = {m[1] for m in TRAVEL_MSGS}
        assert r[0].message.content not in travel_set

    def test_gaming_score_positive(self):
        r = search("gaming setup", INDEX, k=1, model=_mock_for("gaming", 12))
        assert r[0].score > 0.1


# ===== TRAVEL TESTS (12) =====
class TestTravel:
    def test_travel_jeju(self):
        r = search("island vacation Korea", INDEX, k=3, model=_mock_for("travel", 1))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_sightseeing(self):
        r = search("visiting tourist attractions", INDEX, k=3, model=_mock_for("travel", 2))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_accommodation(self):
        r = search("hotel booking reservation", INDEX, k=3, model=_mock_for("travel", 3))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_japan(self):
        r = search("Japan trip sightseeing", INDEX, k=3, model=_mock_for("travel", 4))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_paris(self):
        r = search("European museum visit", INDEX, k=3, model=_mock_for("travel", 5))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_beach(self):
        r = search("tropical beach resort relaxation", INDEX, k=3, model=_mock_for("travel", 6))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_backpacking(self):
        r = search("budget travel adventure", INDEX, k=3, model=_mock_for("travel", 7))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_flights(self):
        r = search("airplane ticket booking cheap", INDEX, k=3, model=_mock_for("travel", 8))
        assert r[0].message.content in [m[1] for m in TRAVEL_MSGS]

    def test_travel_top3_all_travel(self):
        r = search("holiday trip abroad", INDEX, k=3, model=_mock_for("travel", 9))
        travel_set = {m[1] for m in TRAVEL_MSGS}
        assert sum(1 for x in r if x.message.content in travel_set) >= 2

    def test_travel_not_gaming(self):
        r = search("vacation destination", INDEX, k=1, model=_mock_for("travel", 10))
        gaming_set = {m[1] for m in GAMING_MSGS}
        assert r[0].message.content not in gaming_set

    def test_travel_not_study(self):
        r = search("travel itinerary", INDEX, k=1, model=_mock_for("travel", 11))
        study_set = {m[1] for m in STUDY_MSGS}
        assert r[0].message.content not in study_set

    def test_travel_score_positive(self):
        r = search("going abroad", INDEX, k=1, model=_mock_for("travel", 12))
        assert r[0].score > 0.1


# ===== STUDY TESTS (12) =====
class TestStudy:
    def test_study_programming(self):
        r = search("coding assignment help", INDEX, k=3, model=_mock_for("study", 1))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_algorithms(self):
        r = search("data structure practice", INDEX, k=3, model=_mock_for("study", 2))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_math(self):
        r = search("math homework calculus", INDEX, k=3, model=_mock_for("study", 3))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_exam(self):
        r = search("final exam preparation", INDEX, k=3, model=_mock_for("study", 4))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_lab(self):
        r = search("lab report assignment", INDEX, k=3, model=_mock_for("study", 5))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_group(self):
        r = search("study group library meeting", INDEX, k=3, model=_mock_for("study", 6))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_grade(self):
        r = search("test score grade result", INDEX, k=3, model=_mock_for("study", 7))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_deadline(self):
        r = search("paper deadline extension", INDEX, k=3, model=_mock_for("study", 8))
        assert r[0].message.content in [m[1] for m in STUDY_MSGS]

    def test_study_top3_all_study(self):
        r = search("university coursework", INDEX, k=3, model=_mock_for("study", 9))
        study_set = {m[1] for m in STUDY_MSGS}
        assert sum(1 for x in r if x.message.content in study_set) >= 2

    def test_study_not_food(self):
        r = search("homework assignment", INDEX, k=1, model=_mock_for("study", 10))
        food_set = {m[1] for m in FOOD_MSGS}
        assert r[0].message.content not in food_set

    def test_study_not_music(self):
        r = search("exam revision", INDEX, k=1, model=_mock_for("study", 11))
        music_set = {m[1] for m in MUSIC_MSGS}
        assert r[0].message.content not in music_set

    def test_study_score_positive(self):
        r = search("academic work", INDEX, k=1, model=_mock_for("study", 12))
        assert r[0].score > 0.1


# ===== MUSIC TESTS (12) =====
class TestMusic:
    def test_music_concert(self):
        r = search("live concert performance tickets", INDEX, k=3, model=_mock_for("music", 1))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_guitar(self):
        r = search("learning instrument strings", INDEX, k=3, model=_mock_for("music", 2))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_album(self):
        r = search("new song release listen", INDEX, k=3, model=_mock_for("music", 3))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_piano(self):
        r = search("classical piano practice recital", INDEX, k=3, model=_mock_for("music", 4))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_jazz(self):
        r = search("jazz club live music night", INDEX, k=3, model=_mock_for("music", 5))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_streaming(self):
        r = search("music streaming platform playlist", INDEX, k=3, model=_mock_for("music", 6))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_orchestra(self):
        r = search("symphony orchestra performance", INDEX, k=3, model=_mock_for("music", 7))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_production(self):
        r = search("making beats music production", INDEX, k=3, model=_mock_for("music", 8))
        assert r[0].message.content in [m[1] for m in MUSIC_MSGS]

    def test_music_top3_all_music(self):
        r = search("singing and playing", INDEX, k=3, model=_mock_for("music", 9))
        music_set = {m[1] for m in MUSIC_MSGS}
        assert sum(1 for x in r if x.message.content in music_set) >= 2

    def test_music_not_travel(self):
        r = search("band performance", INDEX, k=1, model=_mock_for("music", 10))
        travel_set = {m[1] for m in TRAVEL_MSGS}
        assert r[0].message.content not in travel_set

    def test_music_not_gaming(self):
        r = search("musical instrument", INDEX, k=1, model=_mock_for("music", 11))
        gaming_set = {m[1] for m in GAMING_MSGS}
        assert r[0].message.content not in gaming_set

    def test_music_score_positive(self):
        r = search("melody rhythm song", INDEX, k=1, model=_mock_for("music", 12))
        assert r[0].score > 0.1
