import mysql.connector
from datetime import datetime

# Database connection details (replace with your credentials)
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "report_db"
}

def connector():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM exam_records")
        data = cursor.fetchall()
        formatted_data = []
        for row in data:
    # Convert end_time to datetime object (assuming it's a string at index 8)
            end_time_obj = None  # Adjust format if needed
            if isinstance(row[2], str):
                try:
                    end_time_obj = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S.%f")  # For milliseconds/microseconds
                except ValueError:
                    pass  # Handle parsing error for Option 1 silently (continue to Option 2)

            # Option 2: Truncate string and attempt conversion (if acceptable)
            if end_time_obj is None and row[2] is not None:
                try:
                    end_time_str = row[2][:19]  # Truncate to expected length for "%Y-%m-%d %H:%M:%S"
                    end_time_obj = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"Error parsing end_time (both options) for row: {row}")  # Log parsing errors

            row = list(row)
            if end_time_obj is not None:
                row[2] = end_time_obj.strftime("%Y-%m-%d %H:%M:%S")  # Format without microseconds
            formatted_data.append(row)
        return formatted_data

    except mysql.connector.Error as err:
        print("Error connecting to database:", err)
        return []  # Return empty list on error
    finally:
        if connection:
            connection.close()

# Example usage (can be called from gui.py)
# data = fetch_data()
# for row in data:
#     print(row)
