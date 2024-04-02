import click
from loguru import logger
import pandas as pd
from sqlalchemy import create_engine
from typing import Any

def setup_logger(log_to_file: bool, log_level: str) -> None:
    """
    Configure the logger based on user preferences.

    Args:
    log_to_file (bool): If True, logs are written to a file. If False, logs are printed to the console.
    log_level (str): Level of logging. Example: 'INFO', 'ERROR'.

    Returns:
    None
    """
    if log_to_file:
        logger.add("debug.log", format="{time} {level} {message}", level=log_level.upper())
    else:
        logger.remove()
        logger.add(lambda msg: click.echo(msg, err=True), format="{time} {level} {message}", level=log_level.upper())

def build_db_uri(db_type: str, db_name: str, host: str = None, port: str = None, user: str = None, password: str = None) -> str:
    """
    Build a database URI based on the database type and credentials.

    Args:
    db_type (str): Type of the database ('sqlite' or 'postgresql').
    db_name (str): Name or path of the database.
    host (str, optional): Host of the database (for PostgreSQL).
    port (str, optional): Port of the database (for PostgreSQL).
    user (str, optional): Username for the database (for PostgreSQL).
    password (str, optional): Password for the database (for PostgreSQL).

    Returns:
    str: Database URI.
    """
    if db_type == 'sqlite':
        return f"sqlite:///{db_name}"
    elif db_type == 'postgresql':
        return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    else:
        raise ValueError("Unsupported database type")

def load_csv_to_db(csv_file: str, db_uri: str, table_name: str) -> None:
    """
    Load a CSV file into a database using SQLAlchemy.

    Args:
    csv_file (str): Path to the CSV file.
    db_uri (str): Database URI.
    table_name (str): Name of the table to which the CSV data will be loaded.

    Returns:
    None
    """
    try:
        df = pd.read_csv(csv_file)

        engine = create_engine(db_uri)
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"CSV data from '{csv_file}' successfully loaded into table '{table_name}' in database.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


@click.command()
@click.argument('csv_file', type=str)
@click.argument('db_type', type=click.Choice(['sqlite', 'postgresql'], case_sensitive=False))
@click.argument('db_name', type=str)
@click.argument('table_name', type=str)
@click.option('--host', default='localhost', type=str, help="Database host (for PostgreSQL).")
@click.option('--port', default='5432', type=str, help="Database port (for PostgreSQL).")
@click.option('--user', prompt=True, hide_input=False, default='postgres', type=str, help="Database username (for PostgreSQL).")
@click.option('--password', prompt=True, hide_input=True, type=str, help="Database password (for PostgreSQL).")
@click.option('--log-to-file', is_flag=False, help="Enable logging to a file instead of the console.")
@click.option('--log-level', default='INFO', type=str, help="Set the logging level (e.g., INFO, ERROR).")
def main(csv_file: str, db_type: str, db_name: str, table_name: str, host: str, port: str, user: str, password: str, log_to_file: bool, log_level: str) -> None:
    """
    CLI tool to load data from a CSV file into a SQLite or PostgreSQL database.

    Args:
    csv_file (str): Path to the CSV file.
    db_type (str): Type of the database ('sqlite' or 'postgresql').
    db_name (str): Name or path of the database.
    table_name (str): Name of the table in the database.

    Options:
    --host/--port/--user/--password: Database credentials (for PostgreSQL).
    --log-to-file: If set, logs will be written to a file.
    --log-level: The logging level, such as INFO or ERROR.

    Returns:
    None
    """
    setup_logger(log_to_file, log_level)

    db_uri = build_db_uri(db_type, db_name, host, port, user, password)

    try:
        load_csv_to_db(csv_file, db_uri, table_name)
    except Exception as e:
        click.echo(e)

if __name__ == '__main__':
    main()

