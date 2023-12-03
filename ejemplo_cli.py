import click
from loguru import logger
import math
from typing import Any, Callable

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

def calculate_factorial(number: int) -> int:
    """
    Calculate the factorial of a given number.

    Args:
    number (int): A non-negative integer.

    Returns:
    int: Factorial of the given number.
    """
    if number < 0:
        logger.error(f"Factorial of a negative number {number} is not defined.")
        raise ValueError("Factorial of a negative number is not defined.")
    return math.factorial(number)

@click.command()
@click.argument('number', type=int)
@click.option('--log-to-file', is_flag=True, help="Enable logging to a file instead of the console.")
@click.option('--log-level', default='INFO', type=str, help="Set the logging level (e.g., INFO, ERROR).")
def main(number: int, log_to_file: bool, log_level: str) -> None:
    """
    CLI tool to calculate and print the factorial of a given number.

    Args:
    number (int): A non-negative integer for which factorial is to be calculated.

    Options:
    --log-to-file: If set, logs will be written to a file.
    --log-level: The logging level, such as INFO or ERROR.

    Returns:
    None
    """
    setup_logger(log_to_file, log_level)

    try:
        result = calculate_factorial(number)
        click.echo(f"El factorial de {number} es {result}")
    except ValueError as e:
        click.echo(e)

if __name__ == '__main__':
    main()
