# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2023-12-04

### Added

- .dockerignore file to exclude files and directories from the build context.
- Dockerfile to create a Docker image for the application. To create the python container to execute the postgreSQL database, we use the official image from Docker Hub.
- docker-compose.yml file to define and run multi-container Docker applications. The python and the postgreSQL containers are defined in this file.

## [1.0.1] - 2023-12-07

### Added

- .gitignore file to exclude files and directories from the repository.

### Changed

- docker-compose.yml name of postgres db name.

## [1.0.2] - 2023-12-09

### Added

- Dockerfile.postgres to create a Docker image for the postgreSQL database.
- Dockerfile.python to create a Docker image for the python application.
- init.sql file to create the database and the table for the application.

### Changed

- docker-compose.yml to use the Dockerfile.postgres and Dockerfile.python files.

### Removed

- Dockerfile

## [1.0.3] - 2023-12-09

### Changed

- .gitignore added some extensions to ignore.

## [1.0.3] - 2023-12-10

### Changed

- Dockerfile.python changed cmd prompt to container not shutdown after run the application.

## [1.0.4] - 2024-02-20

### Fixed

- ejemplo_cli.py fixed the error when the user wants the data to be read from a file or a folder of csv files. Also fixed the way of processing the data to avoid errors.

### Changed

- ejemplo_cli.py changed the way of normalizing the data and the way of saving the data using the pandas library instead of the basic use of os library.

## [1.0.4] - 2024-02-22

### Added

- Dockerfile.redis to create a Docker image for the redis database.

### Changed

- ejemplo_cli.py added new functionality to the application. Like basic transformations of the data and the possibility to save the data in a file. Also added the possibility to change the parameters of the operations made in the data.
- docker-compose.yml added the redis container to the application.

## [1.0.5] - 2024-03-03

### Added
- app.py can now successfully connect to the database and load the data from the csv file.

## [1.0.6] - 2024-03-05

### Changed
- app.py added the elimination of outliers in the data.

## [1.0.7] - 2024-03-24

### Added 
- loadandprep.py created file to load and prepare the data.

## [1.0.8] - 2024-04-02

### Changed
- loadandprep.py added the stacionary_and_correlation analysis function to the data loaded.
- loadandprep.py added the create_npz function to the data loaded.
- loadandprep.py added the augmentation_operations function to the npz files.
- loadandprep.py added the data_generator_npz function to the npz files.

## [1.0.9] - 2024-04-15

### Added
- loadandprep.py added the lstm and conv model and the compile_and_fit function.

## [1.0.10] - 2024-05-23

### Changed
- loadandprep.py separated the different functions into different files.

### Added
- test_data.py created file to test the data loaded.
- load_data.py created file to load the data and generate npz.
- train_model.py created file to train the model and save the models generated.
- use_model.py created file to use the model generated and make predictions.