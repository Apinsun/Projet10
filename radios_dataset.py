from PIL import Image
from torch.utils.data import Dataset

class RadiosDataset(Dataset):
    def __init__(self, liste_chemins, transformations):
        # On stocke les chemins
        self.liste_chemins = liste_chemins
        self.transformations = transformations

    def __len__(self):
        # Retourne le nombre total d'images
        return len(self.liste_chemins)

    def __getitem__(self, idx):
        # Cette fonction est appelée automatiquement par le DataLoader
        # Elle charge l'image numéro 'idx' depuis le disque dur
        chemin_fichier = self.liste_chemins[idx]
        
        # .convert('RGB') est une sécurité au cas où une image serait en niveaux de gris pur
        image_pil = Image.open(chemin_fichier).convert('RGB') 
        
        # On applique ton pipeline ResNet
        image_tensor = self.transformations(image_pil)
        
        return image_tensor