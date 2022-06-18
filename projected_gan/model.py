import shutil
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from DatasetClass import ImageDataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generator.FastGan import Generator
from discriminator.discriminator import ProjectedDiscriminator
from pytorch_fid.FID_datasetClass import FIDDataset
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import compute_statistics_of_path
from pytorch_fid.fid_score import calculate_frechet_distance
import numpy as np
import warnings

warnings.simplefilter("ignore", UserWarning)


class ProjectedGan:
    def __init__(self,
                 img_dir: str = "../data/pokemon",
                 image_class: str = "pokemon",
                 batch_size: int = 4,
                 n_th_images: int = 4,
                 num_epochs: int = 100,
                 latent_dim: int = 100,
                 image_size: int = 256,
                 learning_rate: float = 0.0002,
                 log_every_n_th_step: int = 10,
                 log_every_n_th_epoch: int = 2,
                 dims=2048,
                 num_workers=0,
                 start_epoch: int = 0,  # TODO
                 MIME_Type: str = "png",
                 ):

        # Hyperparameters
        self.img_dir = img_dir
        self.image_class = image_class
        self.batch_size = batch_size
        self.n_th_images = n_th_images
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.log_every_n_th_step = log_every_n_th_step
        self.log_every_n_th_epoch = log_every_n_th_epoch
        self.start_epoch = start_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen_data_path = None
        self.dims = dims
        self.num_workers = num_workers
        self.curr_run_folder = None

        # Models
        self.G = Generator().to(self.device)
        self.D = ProjectedDiscriminator().to(self.device)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.fid_device = "cpu"
        self.inception_model = InceptionV3([block_idx]).to(self.fid_device)
        self.inception_model.eval()

        self.m2, self.s2 = self.calc_mu_std_of_real_data()

        self.optimizerG = Adam(self.G.parameters(), lr=learning_rate)
        self.optimizerD = Adam(self.D.discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.dataset = ImageDataset(root_dir=img_dir, transform=transform, RGB=True, MIME_Type=MIME_Type)
        self.train_data = DataLoader(self.dataset, batch_size=batch_size, drop_last=True)


    def create_images(self,
                      gen_dir: str = "gen_data",
                      grid: bool = False,
                      batch: int = 1,
                      n_images: int = 10,
                      model_path: str = "trained_model/ProjGan_pokemon_2022_22_33_44.pth",
                      image_class: str = "pokemon", ):

        weights = torch.load(model_path)
        self.G.load_state_dict(weights)
        self.G.eval()

        gen_dir = Path(gen_dir)
        gen_dir.mkdir(exist_ok=True)

        nrow = 4 if grid else 1
        self.gen_data_path = gen_dir / image_class
        if self.gen_data_path.exists():
            shutil.rmtree(self.gen_data_path)
        self.gen_data_path.mkdir()

        for i in tqdm(range(n_images), desc="Generating new images", total=n_images, leave=True):
            img_name = str(i).zfill(3) + ".png"
            img_path = self.gen_data_path / img_name

            z = torch.randn(batch, self.latent_dim, device=self.device)

            gen_imgs = self.G(z).cpu().detach()
            grid = make_grid(gen_imgs, nrow=nrow)
            grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, min=0.0, max=1.0)
            grid = grid.permute(1, 2, 0)
            grid = grid.numpy()

            plt.imsave(img_path, grid)


    def calc_mu_std_of_real_data(self):
        m2, s2 = compute_statistics_of_path(path=self.img_dir,
                                            model=self.inception_model,
                                            batch_size=50,
                                            dims=self.dims,
                                            device=self.fid_device,
                                            num_workers=self.num_workers)
        return m2, s2


    def create_new_run_folder(self):
        weights_folder = Path(__file__).parent / 'weights'
        runs = sorted(weights_folder.glob('run_*'))
        if len(runs) > 0:
            idx = int(runs[-1].stem.split('_')[1])  # ie. run_023 --> 23
        else:
            idx = -1

        new_run_folder = weights_folder / f'run_{idx+1:03d}'
        new_run_folder.mkdir(exist_ok=False)
        return new_run_folder


    def train(self):
        writer = SummaryWriter()
        images_for_fid = []
        best_fid_score = float("inf")

        self.G.train(True)
        self.D.train(True)
        for epoch in tqdm(range(self.num_epochs), leave=True, total=self.num_epochs, desc="Training"):
            for b_i, (real_imgs, _) in enumerate(self.train_data):
                real_imgs = real_imgs.to(self.device)

                # train discriminator
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)

                with torch.no_grad():
                    gen_imgs = self.G(z)
                # gen_imgs = self.G(z).detach()

                logits_real = self.D(real_imgs)
                logits_fake = self.D(gen_imgs)

                self.optimizerD.zero_grad()
                lossD_fake = (F.relu(torch.ones_like(logits_fake) + logits_fake)).mean()
                lossD_real = (F.relu(torch.ones_like(logits_real) - logits_real)).mean()
                lossD = lossD_real + lossD_fake
                lossD.backward()
                self.optimizerD.step()
                lossD_score = lossD.cpu().detach().numpy()

                if epoch % self.log_every_n_th_epoch == 0 and epoch >= self.start_epoch:
                    images_for_fid.append(gen_imgs)  # each of BxCxHxW

                # train Generator
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.G(z)
                logits_fake = self.D(gen_imgs)
                self.optimizerG.zero_grad()
                lossG = (-logits_fake).mean()
                lossG.backward()
                self.optimizerG.step()

                lossG_score = lossG.cpu().detach().numpy()

                # tensorboard
                if b_i % self.log_every_n_th_step == 0:
                    writer.add_scalar("Discriminator Loss", lossD_score, b_i)
                    writer.add_scalar("Generator loss", lossG_score, b_i)

            if epoch % self.log_every_n_th_epoch == 0:
                grid = make_grid(gen_imgs, nrow=self.n_th_images)
                grid = grid * 0.5 + 0.5
                grid = torch.clamp(grid, min=0.0, max=1.0)
                grid = grid.cpu().detach().numpy()
                writer.add_image(tag="GenImages", img_tensor=grid, global_step=epoch)

            if epoch % self.log_every_n_th_epoch == 0 and epoch >= self.start_epoch:

                start_idx = 0
                images_for_fid = torch.cat(images_for_fid, axis=0)  # NxCxHxW
                fid_dataset = FIDDataset(files=images_for_fid)
                fid_loader = DataLoader(fid_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=self.num_workers)

                pred_arr = np.empty((len(images_for_fid), self.dims))
                for batch in fid_loader:
                    batch = batch.to(self.fid_device)

                    with torch.no_grad():
                        pred = self.inception_model(batch)[0]

                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                    pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                    start_idx = start_idx + pred.shape[0]

                m1 = np.mean(pred_arr, axis=0)
                s1 = np.cov(pred_arr, rowvar=False)

                current_fid_score = calculate_frechet_distance(m1, s1, self.m2, self.s2)

                if current_fid_score < best_fid_score:
                    self.curr_run_folder = self.create_new_run_folder()
                    for p in self.curr_run_folder.glob('G-*'):
                        p.unlink()  # remove all previously saved generator weights
                    torch.save(self.G.state_dict(), self.curr_run_folder / f"G-{epoch}-{current_fid_score}.pth")
                    best_fid_score = current_fid_score

                images_for_fid = []

        writer.close()


if __name__ == "__main__":
    PG = ProjectedGan()
    PG.train()
