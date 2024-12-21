from flask import Flask, render_template, request, jsonify

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import json

from recommendation.recomm import SongRecommendationSystem

# Initialize a SparkSession
#spark = SparkSession.builder.appName("SampleApp").getOrCreate()

spark = SparkSession.builder \
            .appName("SongRecommendationWithArtists") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
# Path to the CSV file
file_path = "./tracks_features.csv"

# Read the CSV file
df = spark.read.csv(file_path, header=True, inferSchema=True)


# Create an instance of the recommendation system
recommender = SongRecommendationSystem(spark)



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    # Get the search query from the URL parameter
    search_query = request.args.get('query', '')

    print(search_query)
    #result = df.filter(col("name").contains(search_query))

    result = df.filter(col("name").rlike(f"(?i){search_query}"))

    # Limit the result to the top 5 rows
    top_5_df = result.limit(5)

    # Convert to JSON format
    json_result = top_5_df.toJSON().collect()

    # Convert JSON strings into Python dictionaries
    parsed_json_result = [json.loads(record) for record in json_result]

    #print(parsed_json_result)
    # Return JSON response
    return jsonify(parsed_json_result)
'''
@app.route('/song', methods=['GET'])
def songrecommendation():

    # Get song recommendations
    given_song_ids = ['12Cbou8Hl4yGGuTZlkLl60' , '0vC9iFuPGIToAw3eI0KdIt']  # Replace with actual song IDs
    recommendations = recommender.get_song_recommendations(given_song_ids, num_recommendations=5)

    print(recommendations)
    
    return recommendations
'''
@app.route('/song', methods=['POST'])
def song_recommendation():
    try:
        # Extract 'track_ids' from the request JSON payload
        data = request.get_json()
        if not data or 'track_ids' not in data:
            return jsonify({"error": "Please provide a 'track_ids' array in the request payload"}), 400
        
        track_ids = data['track_ids']
        
        # Check if track_ids is a list
        if not isinstance(track_ids, list):
            return jsonify({"error": "'track_ids' should be an array of song IDs"}), 400

        # Generate song recommendations
        recommendations = recommender.get_song_recommendations(track_ids, num_recommendations=5)

        print(recommendations)
        
        # Return recommendations as JSON
        return jsonify(json.loads(recommendations)), 200
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969 , debug= True)