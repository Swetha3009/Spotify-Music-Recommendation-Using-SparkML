from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import col, when
import json

class SongRecommendationSystem:
    def __init__(self, spark , model_path = "./model/kmeans_model", file_path = './tracks_features.csv'):
        '''self.spark = SparkSession.builder \
            .appName("SongRecommendationWithArtists") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()'''
        self.spark = spark
        
        self.model_path = model_path
        self.file_path = file_path
        self.model = KMeansModel.load(model_path)
        self.df = self.preprocess_data()

    def preprocess_data(self):
        # Load dataset
        df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)

        # Attributes for recommendation
        attributes = ["danceability", "energy", "liveness", "valence", "tempo"]

        # Replace nulls or non-numeric values and cast to double
        for attr in attributes:
            df = df.withColumn(attr, when(col(attr).isNull(), 0.0).otherwise(col(attr).cast("double")))

        df = df.fillna(0)

        # Encode 'artists' using StringIndexer and OneHotEncoder
        string_indexer = StringIndexer(inputCol="artists", outputCol="artist_index")
        one_hot_encoder = OneHotEncoder(inputCol="artist_index", outputCol="artist_vector")

        # Combine numerical attributes and the artist vector into a single feature vector
        vector_assembler = VectorAssembler(inputCols=attributes + ["artist_vector"], outputCol="features")

        df = string_indexer.fit(df).transform(df)
        df = one_hot_encoder.fit(df).transform(df)
        df = vector_assembler.transform(df)

        # Apply clustering model to the data
        df = self.model.transform(df)
        return df

    def get_song_recommendations(self, given_song_ids, num_recommendations=5):
        recommendations = {}

        # Get clusters for the given songs
        given_songs = self.df.filter(col("id").isin(given_song_ids)).select("id", "cluster")

        recommendations["recommendations"] = []

        # Generate recommendations for each song ID
        for row in given_songs.collect():
            song_id = row["id"]
            cluster_id = row["cluster"]

            # Get similar songs from the same cluster
            similar_songs = self.df.filter((col("cluster") == cluster_id) & (col("id") != song_id)) \
                                   .select("id","name","album", "artists", "danceability", "energy", "liveness", "valence", "tempo") \
                                   .limit(num_recommendations)

            # Collect results and format them as a list of dictionaries

            for song in similar_songs.collect():
                recommendations["recommendations"].append(
                    {
                        "id" : song["id"],
                        "name": song["name"],
                        "album": song["album"],
                        "artists": song["artists"],
                        "danceability": song["danceability"],
                        "energy": song["energy"],
                        "liveness": song["liveness"],
                        "valence": song["valence"],
                        "tempo": song["tempo"],
                    }
                )

        # Convert recommendations to JSON format
        recommendations_json = json.dumps(recommendations, indent=4)
        return recommendations_json
