# Installation Guide for macOS
## Simple Step-by-Step Instructions

This guide will help you install and run "The Culture of International Relations" software on your Mac. Follow each step in order, and don't worry if you're not familiar with the command line - we'll guide you through it!

---

## Step 1: Open Terminal

1. Press `Command + Space` to open Spotlight Search
2. Type "Terminal" and press Enter
3. A window with white or black text will appear - this is your Terminal

💡 **Tip**: Keep Terminal open throughout this installation process.

---

## Step 2: Install Homebrew (The Package Manager)

Homebrew helps install software on your Mac. Copy and paste this command into Terminal and press Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- You may be asked to enter your Mac password (the one you use to log in)
- The installation may take a few minutes
- Follow any instructions that appear on screen

✅ **You'll know it worked when** you see "Installation successful!" or similar message.

---

## Step 3: Install uv (The Python Package Manager)

Copy and paste this command into Terminal and press Enter:

```bash
brew install uv
```

Wait for it to finish (usually less than a minute).

---

## Step 4: Download the Project

Copy and paste these commands **one at a time** into Terminal:

```bash
cd ~
```

Then:

```bash
git clone https://github.com/humlab/the_culture_of_international_relations.git
```

**Note**: If you see a popup asking to install developer tools or Git, click "Install" and wait for it to complete, then run the `git clone` command again.

---

## Step 5: Navigate to the Project

Copy and paste this command:

```bash
cd the_culture_of_international_relations
```

You are now inside the project folder.

---

## Step 6: Set Up the Environment

This step installs Python and all required software. Copy and paste this command:

```bash
uv venv
```

Wait for it to complete, then run:

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the beginning of your command line.

---

## Step 7: Install All Dependencies

Copy and paste this command:

```bash
uv pip install -e .
```

⏱️ **This may take 5-10 minutes** as it downloads and installs many tools. Be patient!

✅ **You'll know it worked when** you see messages about successful installation and no error messages in red.

---

## Step 8: Launch Jupyter Lab

Every time you want to work with the notebooks, follow these steps:

1. Open Terminal
2. Navigate to the project folder:
   ```bash
   cd ~/the_culture_of_international_relations
   ```

3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

4. Start Jupyter Lab:
   ```bash
   uv run jupyter lab
   ```

🎉 **Your web browser will automatically open** showing the Jupyter interface!

---

## Working with Notebooks

Once Jupyter Lab opens in your browser:

1. You'll see a file browser on the left side
2. Double-click on the `notebooks` folder
3. Browse the subfolders:
   - `quantitative_analysis/` - Statistical analysis notebooks
   - `network_analysis/` - Network visualization notebooks
   - `text_analysis/` - Text processing notebooks
4. Double-click any `.ipynb` file to open it
5. Run cells by clicking the "▶️ Run" button or pressing `Shift + Enter`

---

## Stopping Jupyter Lab

When you're done working:

1. In your browser, save your work (File → Save All)
2. Go back to Terminal
3. Press `Control + C` (hold Control, then press C)
4. Type `y` and press Enter if asked to confirm
5. You can now close Terminal

---

## Starting Again Later

Next time you want to use the software:

1. Open Terminal
2. Run these commands:
   ```bash
   cd ~/the_culture_of_international_relations
   source .venv/bin/activate
   uv run jupyter lab
   ```

That's it! Jupyter Lab will open in your browser.

---

## Troubleshooting

### "Command not found" error

If you see "command not found" for any command:
- Make sure you copied the entire command
- Check that previous steps completed successfully
- Try closing Terminal and opening a new one

### Installation seems stuck

If an installation step seems frozen:
- Wait at least 5 minutes (downloads can be slow)
- Check your internet connection
- If truly stuck, press `Control + C` to cancel and try again

### Can't find the project folder

If `cd ~/the_culture_of_international_relations` doesn't work:
- Make sure Step 4 (git clone) completed successfully
- Look for error messages from that step

### Need more help?

Contact your IT support or technical collaborator with:
- A screenshot of the error message
- Which step you're on
- The exact command you tried to run

---

## Summary Checklist

- [ ] Installed Homebrew
- [ ] Installed uv
- [ ] Downloaded the project with `git clone`
- [ ] Created virtual environment with `uv venv`
- [ ] Activated environment with `source .venv/bin/activate`
- [ ] Installed dependencies with `uv pip install -e .`
- [ ] Successfully launched Jupyter Lab
- [ ] Opened and ran a notebook

**Congratulations!** You're ready to start your analysis. 🎓
