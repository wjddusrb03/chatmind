"""Topic tests 06-10: Sports, Movies, Weather, Shopping, Health.

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

SPORTS_MSGS = [
    ("Alex", "Did you watch the Champions League final last night?"),
    ("Kim", "Going to the gym every morning, bench press hit 100kg"),
    ("Lee", "Marathon training schedule is 5km every day this month"),
    ("Park", "NBA playoffs are so intense this year, Lakers vs Celtics"),
    ("Alex", "Tennis match at the local court this Saturday, who's in?"),
    ("Kim", "Swimming 50 laps in the pool today, new personal record"),
    ("Lee", "World Cup qualifiers start next month, can't wait"),
    ("Park", "Yoga class was surprisingly difficult, my body is so stiff"),
]

MOVIES_MSGS = [
    ("Kim", "Movie night this Friday! Let's watch Dune Part 2"),
    ("Park", "I prefer horror movies, how about watching The Exorcist?"),
    ("Alex", "The new Marvel movie was disappointing, bad CGI everywhere"),
    ("Lee", "Just finished watching Parasite again, masterpiece every time"),
    ("Kim", "Oscar nominations are out, Oppenheimer got 13 nominations"),
    ("Park", "Studio Ghibli marathon this weekend, starting with Spirited Away"),
    ("Alex", "Christopher Nolan is the best director of our generation"),
    ("Lee", "Documentary about climate change on Netflix was eye-opening"),
]

WEATHER_MSGS = [
    ("Kim", "Good morning everyone, the weather is really cold today"),
    ("Park", "It's -10 degrees outside, bundle up and stay warm!"),
    ("Alex", "Heavy rain forecast for tomorrow, don't forget your umbrella"),
    ("Lee", "Cherry blossoms are blooming early this year, so beautiful"),
    ("Kim", "Heatwave warning issued, temperature reaching 38 degrees"),
    ("Park", "Snow is falling heavily, roads are very slippery be careful"),
    ("Alex", "Perfect weather for a picnic, sunny with gentle breeze"),
    ("Lee", "Typhoon warning, everyone stay indoors and be safe tonight"),
]

SHOPPING_MSGS = [
    ("Lee", "Black Friday deals are insane, got 70% off on Nike shoes"),
    ("Alex", "New iPhone released today, should I upgrade from 14 to 16?"),
    ("Kim", "Found a vintage leather jacket at the thrift store for $20"),
    ("Park", "Amazon Prime Day starts tomorrow, making my wishlist now"),
    ("Lee", "Grocery shopping tip: buy in bulk at Costco to save money"),
    ("Alex", "Christmas gifts for the family, any budget-friendly ideas?"),
    ("Kim", "The designer bag sale at the department store was amazing"),
    ("Park", "Online shopping addiction is real, my credit card is crying"),
]

HEALTH_MSGS = [
    ("Park", "Got my flu vaccine today, arm is a little sore now"),
    ("Kim", "Trying intermittent fasting 16:8, lost 3kg in a month"),
    ("Lee", "Dentist appointment tomorrow morning, dreading it already"),
    ("Alex", "Started taking vitamin D supplements for the winter season"),
    ("Park", "Allergies are terrible this spring, sneezing non-stop"),
    ("Kim", "Mental health day today, taking a break from everything"),
    ("Lee", "Doctor said my cholesterol is too high, need to change diet"),
    ("Alex", "Meditation app is helping me sleep better, 7 day streak"),
]

TOPIC_MESSAGES = {
    "sports": SPORTS_MSGS,
    "movies": MOVIES_MSGS,
    "weather": WEATHER_MSGS,
    "shopping": SHOPPING_MSGS,
    "health": HEALTH_MSGS,
}


def _build_topic_index():
    messages = []
    vectors = []
    noise_rng = np.random.RandomState(300)

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


# ===== SPORTS TESTS (12) =====
class TestSports:
    def test_sports_soccer(self):
        r = search("football match championship", INDEX, k=3, model=_mock_for("sports", 1))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_gym(self):
        r = search("weightlifting workout fitness", INDEX, k=3, model=_mock_for("sports", 2))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_running(self):
        r = search("running jogging training", INDEX, k=3, model=_mock_for("sports", 3))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_basketball(self):
        r = search("basketball game playoffs", INDEX, k=3, model=_mock_for("sports", 4))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_tennis(self):
        r = search("racket court sports match", INDEX, k=3, model=_mock_for("sports", 5))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_swimming(self):
        r = search("pool swimming laps record", INDEX, k=3, model=_mock_for("sports", 6))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_worldcup(self):
        r = search("international football tournament", INDEX, k=3, model=_mock_for("sports", 7))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_yoga(self):
        r = search("stretching flexibility exercise", INDEX, k=3, model=_mock_for("sports", 8))
        assert r[0].message.content in [m[1] for m in SPORTS_MSGS]

    def test_sports_top3(self):
        r = search("athletic competition", INDEX, k=3, model=_mock_for("sports", 9))
        s = {m[1] for m in SPORTS_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_sports_not_movies(self):
        r = search("sports training", INDEX, k=1, model=_mock_for("sports", 10))
        assert r[0].message.content not in {m[1] for m in MOVIES_MSGS}

    def test_sports_not_shopping(self):
        r = search("exercise routine", INDEX, k=1, model=_mock_for("sports", 11))
        assert r[0].message.content not in {m[1] for m in SHOPPING_MSGS}

    def test_sports_score(self):
        r = search("play sports", INDEX, k=1, model=_mock_for("sports", 12))
        assert r[0].score > 0.1


# ===== MOVIES TESTS (12) =====
class TestMovies:
    def test_movies_scifi(self):
        r = search("science fiction film epic", INDEX, k=3, model=_mock_for("movies", 1))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_horror(self):
        r = search("scary horror thriller film", INDEX, k=3, model=_mock_for("movies", 2))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_superhero(self):
        r = search("superhero comic book movie", INDEX, k=3, model=_mock_for("movies", 3))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_korean(self):
        r = search("Korean film masterpiece award", INDEX, k=3, model=_mock_for("movies", 4))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_oscar(self):
        r = search("Academy Awards nomination ceremony", INDEX, k=3, model=_mock_for("movies", 5))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_anime(self):
        r = search("Japanese animation Studio Ghibli", INDEX, k=3, model=_mock_for("movies", 6))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_director(self):
        r = search("film director cinematography", INDEX, k=3, model=_mock_for("movies", 7))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_documentary(self):
        r = search("documentary series streaming", INDEX, k=3, model=_mock_for("movies", 8))
        assert r[0].message.content in [m[1] for m in MOVIES_MSGS]

    def test_movies_top3(self):
        r = search("watching a film", INDEX, k=3, model=_mock_for("movies", 9))
        s = {m[1] for m in MOVIES_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_movies_not_sports(self):
        r = search("cinema screening", INDEX, k=1, model=_mock_for("movies", 10))
        assert r[0].message.content not in {m[1] for m in SPORTS_MSGS}

    def test_movies_not_weather(self):
        r = search("movie theater", INDEX, k=1, model=_mock_for("movies", 11))
        assert r[0].message.content not in {m[1] for m in WEATHER_MSGS}

    def test_movies_score(self):
        r = search("film and cinema", INDEX, k=1, model=_mock_for("movies", 12))
        assert r[0].score > 0.1


# ===== WEATHER TESTS (12) =====
class TestWeather:
    def test_weather_cold(self):
        r = search("freezing cold temperature", INDEX, k=3, model=_mock_for("weather", 1))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_snow(self):
        r = search("snowing icy roads", INDEX, k=3, model=_mock_for("weather", 2))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_rain(self):
        r = search("rainy day umbrella forecast", INDEX, k=3, model=_mock_for("weather", 3))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_spring(self):
        r = search("flowers blooming spring season", INDEX, k=3, model=_mock_for("weather", 4))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_heat(self):
        r = search("hot summer heatwave", INDEX, k=3, model=_mock_for("weather", 5))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_snow_heavy(self):
        r = search("heavy snowfall slippery", INDEX, k=3, model=_mock_for("weather", 6))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_sunny(self):
        r = search("sunny clear sky outdoor", INDEX, k=3, model=_mock_for("weather", 7))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_typhoon(self):
        r = search("storm hurricane warning danger", INDEX, k=3, model=_mock_for("weather", 8))
        assert r[0].message.content in [m[1] for m in WEATHER_MSGS]

    def test_weather_top3(self):
        r = search("climate conditions today", INDEX, k=3, model=_mock_for("weather", 9))
        s = {m[1] for m in WEATHER_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_weather_not_health(self):
        r = search("temperature outside", INDEX, k=1, model=_mock_for("weather", 10))
        assert r[0].message.content not in {m[1] for m in HEALTH_MSGS}

    def test_weather_not_shopping(self):
        r = search("forecast tomorrow", INDEX, k=1, model=_mock_for("weather", 11))
        assert r[0].message.content not in {m[1] for m in SHOPPING_MSGS}

    def test_weather_score(self):
        r = search("weather report", INDEX, k=1, model=_mock_for("weather", 12))
        assert r[0].score > 0.1


# ===== SHOPPING TESTS (12) =====
class TestShopping:
    def test_shopping_sale(self):
        r = search("discount sale promotion", INDEX, k=3, model=_mock_for("shopping", 1))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_phone(self):
        r = search("smartphone new model upgrade", INDEX, k=3, model=_mock_for("shopping", 2))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_thrift(self):
        r = search("secondhand vintage cheap clothes", INDEX, k=3, model=_mock_for("shopping", 3))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_online(self):
        r = search("online shopping delivery order", INDEX, k=3, model=_mock_for("shopping", 4))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_grocery(self):
        r = search("supermarket groceries bulk buy", INDEX, k=3, model=_mock_for("shopping", 5))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_gifts(self):
        r = search("present gift birthday Christmas", INDEX, k=3, model=_mock_for("shopping", 6))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_luxury(self):
        r = search("designer brand luxury fashion", INDEX, k=3, model=_mock_for("shopping", 7))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_spending(self):
        r = search("spending money credit card bills", INDEX, k=3, model=_mock_for("shopping", 8))
        assert r[0].message.content in [m[1] for m in SHOPPING_MSGS]

    def test_shopping_top3(self):
        r = search("buying things store", INDEX, k=3, model=_mock_for("shopping", 9))
        s = {m[1] for m in SHOPPING_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_shopping_not_sports(self):
        r = search("purchase items", INDEX, k=1, model=_mock_for("shopping", 10))
        assert r[0].message.content not in {m[1] for m in SPORTS_MSGS}

    def test_shopping_not_weather(self):
        r = search("shopping cart checkout", INDEX, k=1, model=_mock_for("shopping", 11))
        assert r[0].message.content not in {m[1] for m in WEATHER_MSGS}

    def test_shopping_score(self):
        r = search("retail store mall", INDEX, k=1, model=_mock_for("shopping", 12))
        assert r[0].score > 0.1


# ===== HEALTH TESTS (12) =====
class TestHealth:
    def test_health_vaccine(self):
        r = search("vaccination shot immunity", INDEX, k=3, model=_mock_for("health", 1))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_diet(self):
        r = search("fasting weight loss diet", INDEX, k=3, model=_mock_for("health", 2))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_dentist(self):
        r = search("dental checkup appointment teeth", INDEX, k=3, model=_mock_for("health", 3))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_vitamins(self):
        r = search("supplement nutrition vitamins", INDEX, k=3, model=_mock_for("health", 4))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_allergies(self):
        r = search("allergy symptoms sneezing", INDEX, k=3, model=_mock_for("health", 5))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_mental(self):
        r = search("mental wellness stress break", INDEX, k=3, model=_mock_for("health", 6))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_cholesterol(self):
        r = search("blood test medical checkup results", INDEX, k=3, model=_mock_for("health", 7))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_sleep(self):
        r = search("insomnia sleeping better meditation", INDEX, k=3, model=_mock_for("health", 8))
        assert r[0].message.content in [m[1] for m in HEALTH_MSGS]

    def test_health_top3(self):
        r = search("medical health wellness", INDEX, k=3, model=_mock_for("health", 9))
        s = {m[1] for m in HEALTH_MSGS}
        assert sum(1 for x in r if x.message.content in s) >= 2

    def test_health_not_movies(self):
        r = search("doctor hospital", INDEX, k=1, model=_mock_for("health", 10))
        assert r[0].message.content not in {m[1] for m in MOVIES_MSGS}

    def test_health_not_sports(self):
        r = search("medical treatment recovery", INDEX, k=1, model=_mock_for("health", 11))
        assert r[0].message.content not in {m[1] for m in SPORTS_MSGS}

    def test_health_score(self):
        r = search("healthcare clinic", INDEX, k=1, model=_mock_for("health", 12))
        assert r[0].score > 0.1
