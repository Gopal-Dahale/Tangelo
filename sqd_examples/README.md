# Tangelo SQD

This repository contains a collection of Jupyter notebooks and resources demonstrating the use of Tangelo for quantum chemistry simulations. The examples focus on Sample-based Quantum Diagonalization (SQD), Density Matrix Embedding Theory (DMET), and the QM/MM method for solving complex molecular problems.

Checkout the **`Report.pdf`** for more information and results obtained with simulations. Refer to **`Presentation.mkv`** for the video presentation. [Google Drive Link](https://drive.google.com/file/d/1uMRNehYvUtgddLNCAZPCtnI401FCg-is/view?usp=sharing) to the presentation.

## Project Structure

### Notebooks

1. **`1_sqd.ipynb`**
   - Demonstrates the SQD algorithm for approximating the ground state of the nitrogen molecule.
   - Explains quantum chemistry concepts like second quantization and molecular Hamiltonians.

2. **`2_dmet_sqd.ipynb`**
   - Introduces DMET for decomposing molecular systems into fragments and environments.
   - Includes examples like DMET-CCSD on butane and DMET-VQE on a hydrogen ring.

3. **`3_cyclohexane_hf.ipynb`**
   - Analyzes conformations of cyclohexane using Hartree-Fock calculations.
   - Reads molecular data from `cyclohexane_chair-flip.xyz`.
   - Saves energy calculations in `cyclohexane_hf_energies.npy`.

4. **`4_dmet_cyclohexane.ipynb`**
   - Applies DMET to cyclohexane conformations.
   - Uses Tangelo for problem decomposition and quantum chemistry calculations.

5. **`5_qmmm.ipynb`**
   - Explores the QM/MM method for the solvation of glycine in water.
   - Combines quantum mechanics and molecular mechanics to model complex systems.
   - Calculates solvation energies using Tangelo's tools and the SQD algorithm.

### Additional Files

- **`requirements.txt`**: Python dependencies.
- **`cyclohexane_chair-flip.xyz`**: Contains molecular data for cyclohexane conformations.

## How to Use

1. Environment creation (with [uv](https://github.com/astral-sh/uv)). You can also use PyEnv, but I prefer uv as the installation time is lower.
```bash
uv venv my-env
source my-env/bin/activate
```

2. Clone and set up
```bash
git clone https://github.com/Gopal-Dahale/Tangelo.git
cd Tangelo
git pull origin sqd
git checkout sqd
uv pip install -e .
```

Note: Prefix pip installation with `uv` i.e., `uv pip install <package-name>` or `uv pip install -r requirements.txt` to ensure packages get installed inside the environment.

- SQD uses [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd) and [`qiskit-addon-dice-solver`](https://github.com/Qiskit/qiskit-addon-dice-solver). Follow the instructions to install them from the respective repositories.
- Remaining helper packages can be installed with `uv pip install -r requirements.txt`. The `requirements.txt` is present in the current level.

3. Run `jupyter notebook` (needs separate installation `uv pip install jupyter`) 

4. Follow the instructions in each notebook to run the examples.

## References

- [Tangelo Documentation](https://sandbox-quantum.github.io/Tangelo-Examples/)
- [Towards quantum-centric simulations of extended molecules: sample-based quantum diagonalization enhanced with density matrix embedding theory](https://arxiv.org/abs/2411.09861)
