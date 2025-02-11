## Information about cases:

Son 3 carpetas: easy, medium and hard

- En easy hay 1 submapa, el objetivo es emparejar las imágenes entre sí. 
- En medium hay 3 submapas de la misma secuencia, el objetivo es emparejar submapas de un submapa contra imágenes de otros submapas. 
- En hard hay 1 submapa de otra secuencia, el objetivo es emparejar submapas de medium contra él (misma idea que el trabajo de Computer Vision)

## **📌 Workflow for Your Git Operations**

### **1️⃣ Pulling & Pushing in Your Main Repository (`colon_matching`)**

#### ✅ **Pull the latest changes from your repo (`colon_matching`)**

```bash
git pull origin main
```

#### ✅ **Push your changes to your repo (`colon_matching`)**

```bash
git add .
git commit -m "Updated main repository"
git push origin main
```

---

### **2️⃣ Pulling & Pushing in Your Submodule (`image-matching_models`)**

#### ✅ **Move into the submodule**

```bash
cd utils/image-matching_models
```

#### ✅ **Pull the latest changes from your fork (`origin`)**

```bash
git pull origin main
```

#### ✅ **Pull updates from the original upstream repo (`alexstoken`)**

If the upstream repo (`alexstoken/image-matching-models`) has updates you want:

```bash
git fetch upstream
git merge upstream/main  # or rebase: git rebase upstream/main
```

#### ✅ **Push the updates to your fork (`origin`)**

```bash
git push origin main
```

#### ✅ **Move back to the main repo**

```bash
cd ../..
```

#### ✅ **Update the submodule reference in your main repo**

If the submodule (`image-matching_models`) was updated, your main repo (`colon_matching`) needs to track the new commit:

```bash
git add utils/image-matching_models
git commit -m "Updated submodule reference"
git push origin main
```

---

### **3️⃣ Cloning the Repo & Submodules on a New Machine**

#### ✅ **Clone your main repository (including submodules)**

```bash
git clone --recurse-submodules https://github.com/ipastore/colon_matching.git
```

_(This ensures the submodule is downloaded too.)_

#### ✅ **If you already cloned it without submodules**

```bash
git submodule update --init --recursive
```

#### ✅ **To pull the latest updates for the submodule**

```bash
git submodule update --remote --merge
```

---

### **🔹 Summary Table: Git Commands for Each Repository**

|Action|**Main Repo (`colon_matching`)**|**Submodule (`image-matching_models`)**|
|---|---|---|
|**Pull latest changes**|`git pull origin main`|`cd utils/image-matching_models && git pull origin main`|
|**Pull from upstream (original repo of fork)**|_Not applicable_|`git fetch upstream && git merge upstream/main`|
|**Push changes**|`git push origin main`|`git push origin main`|
|**Update submodule reference in main repo**|`git add utils/image-matching_models && git commit -m "Updated submodule"`|_Not needed_|
|**Clone repo with submodules**|`git clone --recurse-submodules <repo-url>`|_Handled automatically_|
|**Manually update submodules**|`git submodule update --init --recursive`|`git submodule update --remote --merge`|

---

