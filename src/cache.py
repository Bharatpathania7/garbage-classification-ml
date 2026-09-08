import os
import json

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(
    CACHE_DIR,
    "predictions.json"
)
os.makedirs(
    CACHE_DIR,
    exist_ok=True
)
def load_cache():
    """
    Load existing cache from JSON file.
    """

    if not os.path.exists(CACHE_FILE):
        return {}

    with open(
        CACHE_FILE,
        "r"
    ) as file:

        return json.load(file)

def save_cache(cache):
    """
    Save cache dictionary to JSON file.
    """

    with open(
        CACHE_FILE,
        "w"
    ) as file:

        json.dump(
            cache,
            file,
            indent=4
        )
def get_cached_result(fingerprint):
    """
    Return cached prediction if fingerprint exists.
    """

    cache = load_cache()

    return cache.get(fingerprint)
def store_result(
    fingerprint,
    prediction,
    confidence
):
    """
    Store prediction result using image fingerprint.
    """

    cache = load_cache()

    cache[fingerprint] = {
        "prediction": prediction,
        "confidence": confidence
    }

    save_cache(cache)
if __name__ == "__main__":

    print("=" * 50)
    print("CACHE TEST")
    print("=" * 50)

    test_fingerprint = (
        "5379fa50f715756ea408075b56965e55"
        "ea29fbdf991cc3f077e781a4ecd41100"
    )

    # Check cache
    result = get_cached_result(
        test_fingerprint
    )

    if result is None:

        print("\nCache MISS ❌")

        
        prediction = "cardboard"
        confidence = 0.95

        
        store_result(
            test_fingerprint,
            prediction,
            confidence
        )

        print("Prediction stored in cache.")

    else:

        print("\nCache HIT ✅")

        print("Cached result:")
        print(result)