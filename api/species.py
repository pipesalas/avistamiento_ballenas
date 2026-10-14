import re
import unicodedata

PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#882255",
    "#44AA99",
    "#117733",
    "#332288",
    "#AA4499",
    "#88CCEE",
    "#661100",
    "#999933",
    "#6699CC",
]

# Substrings are matched against a lowercase filename. Longer keys win.
PHOTO_HINTS = [
    ("lobo fino", "Lobo fino austral"),
    ("delfin chileno", "Delfín chileno"),
    ("elfin chileno", "Delfín chileno"),
    ("chungu", "Chungungo"),
    ("calder", "Calderón gris"),
    ("jorobada", "Ballena jorobada"),
    ("franca", "Ballena franca austral"),
    ("orca", "Orca"),
    ("sei", "Ballena sei"),
    ("azul", "Ballena azul"),
    ("austral", "Delfín austral"),
    ("fino", "Lobo fino austral"),
    ("fin", "Ballena fin"),
]


SPECIES_FIXES = {
    "Chungungo (Contra felina)": "Chungungo (Lontra felina)",
    "Chungungo (Lontra felina": "Chungungo (Lontra felina)",
    "Lobo comun (Otaria flavescens)": "Lobo común (Otaria flavescens)",
    "Lobo común (Otaria flavences)": "Lobo común (Otaria flavescens)",
    "Ballena jorobada (Megaptera noveangliae)": "Ballena jorobada (Megaptera novaeangliae)",
    "Ballena jorobada (Megaptera novaengliae)": "Ballena jorobada (Megaptera novaeangliae)",
    "Lobo fino de Juan fernández (Arctocephalus philippii)": "Lobo fino de Juan Fernández (Arctocephalus philippii)",
}

BEHAVIOR_FIXES = {
    "alimentandose": "Alimentándose",
    "alimentándose": "Alimentándose",
    "alimetandose": "Alimentándose",
    "alimentadose": "Alimentándose",
    "forrajeo": "Forrajeo",
    "forrajeando": "Forrajeo",
    "forrajendo": "Forrajeo",
    "navegando": "Navegando",
    "navagendo": "Navegando",
    "descansando": "Descansando",
    "descanzando": "Descansando",
    "descansando sobre la roca": "Descansando sobre la roca",
    "desplazandose": "Desplazándose",
    "desplazándose": "Desplazándose",
    "sociabilizando": "Socializando",
    "socializando": "Socializando",
    "social": "Socializando",
    "práctica social": "Socializando",
    "criando": "Criando",
    "amamantando": "Amamantando",
    "jugando": "Jugando",
    "saltando": "Saltando",
    "termoregulando": "Termorregulando",
    "muerta": "Muerto / Varado",
    "muerto": "Muerto / Varado",
}


def normalize_species(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name).strip())
    text = text.replace("novaengliae", "novaeangliae").replace("noveangliae", "novaeangliae")
    return SPECIES_FIXES.get(text, text)


def short_name(name: str) -> str:
    return normalize_species(name).split(" (")[0]


def slugify(name: str) -> str:
    text = short_name(name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def normalize_behavior(value) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() == "nan":
        return None
    return BEHAVIOR_FIXES.get(text.lower(), text)


def species_from_filename(filename: str) -> str | None:
    folded = filename.lower().replace("_", " ").replace("-", " ").replace(":", " ")
    for needle, label in PHOTO_HINTS:
        if needle not in folded:
            continue
        if needle == "fin" and re.search(r"delfin|fino", folded):
            continue
        return label
    return None
