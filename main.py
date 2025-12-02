# import necessary libraries
import hydra

from src import BiplanarReg, ISLES2024Dataset

@hydra.main(version_base=None, config_path='configs')
def main(config):
    # get data
    dataset = ISLES2024Dataset(config.data)
    id_dict = dataset[config.data.index]

    optimizer = BiplanarReg(config.model, id_dict)
    optimizer.fit()

if __name__ == "__main__":
    main()