from collections import defaultdict
import numpy as np
import numpy.typing as npt
import io
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Geometry
from PIL import Image as image
from typing import Dict, Sequence, List, Optional, Union, Tuple
from compchemkit.fingerprints import AtomEnvironment
from compchemkit.fingerprints import _MorganFingerprint
from compchemkit.utils.custom_types import RNGATuple


def shap2atomweight(
    mol: Chem.Mol, fingerprint: _MorganFingerprint, shap_mat: npt.NDArray[np.float_]
) -> list[float]:
    bit_atom_env_dict: dict[int, Sequence[AtomEnvironment]]
    bit_atom_env_dict = dict(
        fingerprint.bit2atom_mapping(mol)
    )  # MyPy invariants make me do this.
    atom_weight_dict = assign_prediction_importance(bit_atom_env_dict, shap_mat)
    atom_weight_list = [
        atom_weight_dict[a_idx] if a_idx in atom_weight_dict else 0
        for a_idx in range(mol.GetNumAtoms())
    ]
    return atom_weight_list


def assign_prediction_importance(
    bit_dict: Dict[int, Sequence[AtomEnvironment]], weights: npt.NDArray[np.float_]
) -> Dict[int, float]:
    atom_contribution: Dict[int, float] = defaultdict(lambda: 0)
    for bit, atom_env_list in bit_dict.items():  # type: int, Sequence[AtomEnvironment]
        n_machtes = len(atom_env_list)
        for atom_set in atom_env_list:
            for atom in atom_set.environment_atoms:
                atom_contribution[atom] += weights[bit] / (
                    len(atom_set.environment_atoms) * n_machtes
                )
    assert np.isclose(sum(weights), sum(atom_contribution.values())), (
        sum(weights),
        sum(atom_contribution.values()),
    )
    return atom_contribution


def get_similaritymap_from_weights(
    mol: Chem.Mol,
    weights: Union[npt.NDArray[np.float_], List[float], Tuple[float]],
    draw2d: Draw.MolDraw2DCairo,
    sigma: Optional[float] = None,
    sigma_f: float = 0.3,
    contour_lines: int = 10,
    contour_params: Optional[Draw.ContourParams] = None,
) -> Draw.MolDraw2D:
    """Generates the similarity map for a molecule given the atomic weights.
     Stolen... uhm... copied from Chem.Draw.SimilarityMaps

    Parameters
    ----------
    mol: Chem.Mol
        the molecule of interest.
    weights: Union[npt.NDArray[np.float_], List[float], Tuple[float]]
    draw2d: Draw.MolDraw2DCairo
    sigma: Optional[float]
    sigma_f: float
    contour_lines: int
    contour_params: Optional[Draw.ContourParams]

    Returns
    -------
    Draw.MolDraw2D
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    if not mol.GetNumConformers():
        Draw.rdDepictor.Compute2DCoords(mol)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = (
                sigma_f
                * (
                    mol.GetConformer().GetAtomPosition(idx1)
                    - mol.GetConformer().GetAtomPosition(idx2)
                ).Length()
            )
        else:
            sigma = (
                sigma_f
                * (
                    mol.GetConformer().GetAtomPosition(0)
                    - mol.GetConformer().GetAtomPosition(1)
                ).Length()
            )
        sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
        atom_pos = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(atom_pos.x, atom_pos.y))
    draw2d.DrawMolecule(mol)
    draw2d.ClearDrawing()
    if not contour_params:
        contour_params = Draw.ContourParams()
        contour_params.fillGrid = True
        contour_params.gridResolution = 0.1
        contour_params.extraGridPadding = 0.5
    Draw.ContourAndDrawGaussians(
        draw2d, locs, weights, sigmas, nContours=contour_lines, params=contour_params
    )
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d


def rdkit_gaussplot(
    mol: Chem.Mol,
    weights: npt.NDArray[np.float_],
    n_contour_lines: int = 5,
    color_tuple: Optional[Tuple[RNGATuple, RNGATuple, RNGATuple]] = None,
) -> Draw.MolDraw2D:
    d = Draw.MolDraw2DCairo(600, 600)
    # Coloring atoms of element 0 to 100 black
    d.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    cps = Draw.ContourParams()
    cps.fillGrid = True
    cps.gridResolution = 0.02
    cps.extraGridPadding = 1.2
    coolwarm = ((0.017, 0.50, 0.850, 0.5), (1.0, 1.0, 1.0, 0.5), (1.0, 0.25, 0.0, 0.5))

    if color_tuple is None:
        color_tuple = coolwarm

    cps.setColourMap(color_tuple)

    d = get_similaritymap_from_weights(
        mol,
        weights,
        contour_lines=n_contour_lines,
        draw2d=d,
        contour_params=cps,
        sigma_f=0.4,
    )
    d.FinishDrawing()
    return d


def show_png(data: bytes) -> image.Image:
    """Transform bytes to Image.

    Parameters
    ----------
    data: bytes
        Image bytes.

    Returns
    -------
    image.Image
        An PIL.Image.Image object.
    """
    bio = io.BytesIO(data)
    img = image.open(bio)
    return img
