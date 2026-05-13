from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class RadiosDataset(Dataset):
    def __init__(self, chemins_images, labels=None, transformations=None):
        
        # --- 1. L'AUTO-DÉTECTION (Fini l'erreur PosixPath !) ---
        # Si on donne juste un chemin de dossier, la classe cherche les images toute seule
        if isinstance(chemins_images, (str, Path)) and Path(chemins_images).is_dir():
            self.chemins = [fichier for fichier in Path(chemins_images).iterdir() if fichier.is_file()]
        else:
            # Si c'est déjà une liste d'images, on la garde telle quelle
            self.chemins = list(chemins_images)
            
        self.labels = labels
        self.transformations = transformations

        # --- 2. LE FILET DE SÉCURITÉ ---
        # On s'assure qu'on n'a pas décalé nos listes (très important en Deep Learning)
        if self.labels is not None:
            if len(self.chemins) != len(self.labels):
                raise ValueError(f"CRASH ÉVITÉ : Vous avez {len(self.chemins)} images mais {len(self.labels)} labels !")

    def __len__(self):
        return len(self.chemins)

    def __getitem__(self, idx):
        chemin = self.chemins[idx]
        image_pil = Image.open(chemin).convert('RGB')
        
        if self.transformations:
            image_tensor = self.transformations(image_pil)
        else:
            image_tensor = image_pil
            
        if self.labels is not None:
            label = self.labels[idx]
            return image_tensor, label
        else:
            return image_tensor