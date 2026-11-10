"""Run sentiment inference for the Harbor House Café reviews."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_with_sentiment.csv"
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
SENTIMENT_BANDS = ("positive", "neutral", "negative")


def band_from_stars(stars: int) -> str:
	"""Map star ratings to the project's fixed sentiment bands."""
	if stars <= 2:
		return "negative"
	if stars == 3:
		return "neutral"
	return "positive"


def load_reviews(path: Path = INPUT_PATH) -> pd.DataFrame:
	"""Load and validate the input review data without silently dropping rows."""
	if not path.is_file():
		raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

	reviews = pd.read_csv(path)
	required_columns = {"review_id", "rating", "review_text"}
	missing_columns = required_columns.difference(reviews.columns)
	if missing_columns:
		raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
	if reviews.empty:
		raise ValueError("El archivo no contiene reseñas.")
	if reviews["review_text"].isna().any():
		raise ValueError("Hay reseñas sin texto; corrige esos datos antes de inferir.")

	reviews["rating"] = pd.to_numeric(reviews["rating"], errors="raise")
	if not reviews["rating"].between(1, 5).all():
		raise ValueError("Todos los ratings deben estar entre 1 y 5.")
	reviews["review_text"] = reviews["review_text"].astype(str)
	if reviews["review_text"].str.strip().eq("").any():
		raise ValueError("Hay reseñas con texto vacío; corrige esos datos antes de inferir.")
	return reviews


def predict_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
	"""Load the fixed model once and append one prediction per review."""
	from transformers import pipeline

	sentiment_model = pipeline(
		"sentiment-analysis",
		model=MODEL_ID,
		tokenizer=MODEL_ID,
		device=-1,
	)
	predictions = sentiment_model(
		reviews["review_text"].tolist(),
		batch_size=16,
		truncation=True,
	)

	if len(predictions) != len(reviews):
		raise RuntimeError("El número de predicciones no coincide con el de reseñas.")

	# The fixed model returns labels such as "1 star" or "5 stars".
	predicted_stars = []
	for prediction in predictions:
		match = re.search(r"\d+", str(prediction["label"]))
		if match is None:
			raise ValueError(f"Etiqueta de estrellas inesperada: {prediction['label']!r}")
		stars = int(match.group())
		if stars not in range(1, 6):
			raise ValueError(f"Predicción fuera del rango 1-5: {stars}")
		predicted_stars.append(stars)

	enriched = reviews.copy()
	enriched["predicted_stars"] = predicted_stars
	enriched["prediction_confidence"] = [float(item["score"]) for item in predictions]
	enriched["sentiment_band"] = enriched["predicted_stars"].map(band_from_stars)
	enriched["human_rating_band"] = enriched["rating"].astype(int).map(band_from_stars)
	return enriched


def main() -> None:
	reviews = load_reviews()
	enriched = predict_reviews(reviews)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(OUTPUT_PATH, index=False)

	predicted_share = (
		enriched["sentiment_band"].value_counts(normalize=True)
		.reindex(SENTIMENT_BANDS, fill_value=0)
		.mul(100)
	)
	human_share = (
		enriched["human_rating_band"].value_counts(normalize=True)
		.reindex(SENTIMENT_BANDS, fill_value=0)
		.mul(100)
	)
	false_negatives = enriched[
		enriched["predicted_stars"].le(2) & enriched["rating"].ge(4)
	]

	print(f"Modelo: {MODEL_ID}")
	print(f"Reseñas procesadas: {len(enriched)}")
	print(f"Rating humano promedio: {enriched['rating'].mean():.2f} / 5")
	print(f"Estrellas predichas promedio: {enriched['predicted_stars'].mean():.2f} / 5")
	print("\nParticipación por banda (%):")
	print(
		pd.DataFrame(
			{
				"rating humano": human_share,
				"predicción del modelo": predicted_share,
				"diferencia (pp)": predicted_share - human_share,
			}
		).round(1).to_string()
	)
	print(f"\nFalsos negativos (rating humano 4-5, predicción 1-2): {len(false_negatives)}")
	print(f"Archivo enriquecido: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
