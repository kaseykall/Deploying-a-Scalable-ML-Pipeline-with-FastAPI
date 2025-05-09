import pickle

# Paths to your saved artifacts
model_path = "model/model.pkl"
encoder_path = "model/encoder.pkl"
lb_path = "model/lb.pkl"

def test_artifact_loading():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ model.pkl loaded successfully")

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        print("✅ encoder.pkl loaded successfully")

        with open(lb_path, "rb") as f:
            lb = pickle.load(f)
        print("✅ lb.pkl loaded successfully")

    except Exception as e:
        print("❌ Error loading artifacts:", e)

if __name__ == "__main__":
    test_artifact_loading()
