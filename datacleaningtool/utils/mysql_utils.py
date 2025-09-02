# In mysql_utils.py

from sqlalchemy import create_engine
import pandas as pd


def get_mysql_connection(host, user, password, database):
    # Deze functie blijft zoals hij was
    try:
        connection_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        engine = create_engine(connection_str)
        return engine.connect()
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")


def fetch_data(connection, table_name_or_query, limit=10000, is_query=False):
    """
    Haalt gegevens op uit een databasetabel of door een aangepaste query uit te voeren.

    Parameters:
        connection: Actieve databaseverbinding
        table_name_or_query: Naam van de tabel om op te vragen OF de volledige SQL-querystring
        limit: Maximaal aantal rijen om op te halen (alleen van toepassing als het geen aangepaste query is en is_query False is)
        is_query: Boolean, True als table_name_or_query een volledige query is, False als het een tabelnaam is

    Returns:
        Pandas DataFrame met de queryresultaten
    """
    query_to_execute = None  # Initialiseer hier de variabele
    try:
        if is_query:
            query_to_execute = table_name_or_query
        else:
            query_to_execute = f"SELECT * FROM {table_name_or_query} LIMIT {limit}"

        # Als query_to_execute hier None is (wat niet zou moeten gebeuren met de if/else),
        # dan zou pd.read_sql waarschijnlijk een fout geven.
        if query_to_execute is None:
            # Dit is een extra veiligheidscheck, zou niet nodig moeten zijn met de huidige logica
            raise ValueError("Query string could not be determined.")

        return pd.read_sql(query_to_execute, connection)
    except Exception as e:
        # Maak de foutmelding iets robuuster voor het geval query_to_execute onbekend is
        error_query_context = query_to_execute if query_to_execute is not None else "unavailable"
        raise Exception(f"Data retrieval failed for query '{error_query_context}': {e}")