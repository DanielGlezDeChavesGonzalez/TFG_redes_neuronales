import click


def load_best_model ():
    
    # checkpoint_filepath_lstm = f'./weights/model_lstm_{model_version}_dataset_{dataset_version}_{{loss:.3f}}.weights.h5'
    
    return None
    

@click.command()

def main() -> None:
    
    best_model = load_best_model()