import psycopg2
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DB_CONFIG = {
    "dbname": "pgpu",
    "user": "ec2-user",
    #"password": "your_password",
    "host": "localhost",
    "port": "28818"
}

NAME="faiss"

VECS_TABLE="test_vecs_normalized_large"
#VECS_TABLE="test_vecs_small"
CENTROIDS_TABLE=VECS_TABLE+"_centroids"
OUTPUT_FILE = VECS_TABLE+"_"+NAME+".png"

def get_data(table_name, col_name):
    """Connects to DB and fetches vector data from a specific table."""
    conn = None
    vectors = []

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # We cast to text to ensure easy parsing in Python
        # regardless of whether the pgvector python adapter is registered.
        cur.execute(f"SELECT {col_name}::text FROM {table_name}")
        rows = cur.fetchall()

        for row in rows:
            # Row is typically a string like "[0.123, -0.456]"
            # We strip brackets and split by comma
            clean_str = row[0].strip('[]')
            x, y = map(float, clean_str.split(','))
            vectors.append((x, y))

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error connecting to database: {error}")
    finally:
        if conn is not None:
            conn.close()

    return vectors

def plot_vectors(data, centroids=None):
    """Generates the scatter plot."""
    if not data:
        print("No data found to plot.")
        return

    # Unzip the list of (x, y) tuples into two lists
    x_vals, y_vals = zip(*data)

    plt.figure(figsize=(8, 8))

    # Create the scatter plot
    plt.scatter(x_vals, y_vals, alpha=0.6, edgecolors='b', label='Vectors')

    # Plot centroids if they exist
    if centroids:
        cx, cy = zip(*centroids)
        # Plot red 'X' markers for centroids
        plt.scatter(cx, cy, color='red', marker='x', s=100, linewidth=2, label='Centroids')

    # Draw a unit circle for reference (since vectors are normalized)
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.5, label='Unit Circle')
    plt.gca().add_patch(unit_circle)

    # Set chart aesthetics
    plt.title('2D Vector Distribution - '+VECS_TABLE+"_"+NAME)
    plt.xlabel('Dimension 1 (X)')
    plt.ylabel('Dimension 2 (Y)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # Ensure the aspect ratio is square so the circle looks like a circle
    plt.axis('equal')

    # Save to file
    plt.savefig(OUTPUT_FILE)
    print(f"Success! Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    vectors = get_data(VECS_TABLE, "embedding")
    centroids = get_data(CENTROIDS_TABLE, "vector")
    plot_vectors(vectors, centroids)