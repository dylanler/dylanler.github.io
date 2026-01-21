# Claude Code Instructions for dylanler.github.io

## Project Overview

This is Dylan Ler's personal technical blog built with Hugo and the PaperMod theme. It's deployed to GitHub Pages at https://dylanler.github.io/.

**Topics covered**: AI/ML research, synthetic data, LLMs, startups, entrepreneurship, leadership, and team management.

## Directory Structure

```
dylanler.github.io/
├── dylan-blog/                 # Main Hugo site
│   ├── content/posts/          # Published blog posts (edit here)
│   ├── content/images/         # Image assets
│   ├── public/                 # Generated output (DO NOT edit)
│   ├── themes/PaperMod/        # Theme (git submodule)
│   ├── archetypes/             # Post templates
│   └── hugo.toml               # Site configuration
├── drafts/                     # Research notes & draft ideas
└── .github/workflows/hugo.yml  # CI/CD deployment
```

## Key Commands

### Local Development
```bash
cd dylan-blog
hugo serve                # Start dev server at http://localhost:1313
hugo serve -D             # Include draft posts
hugo                      # Build site to ./public
hugo --minify             # Production build
```

### Creating a New Post
```bash
cd dylan-blog
hugo new posts/my-new-post.md
```
This creates a post with `draft = true`. Set `draft = false` to publish.

### Deployment
Push to `main` branch → GitHub Actions automatically builds and deploys.

## Post Frontmatter Format

Posts use TOML frontmatter (with `+++` delimiters):

```toml
+++
title = 'Post Title Here'
date = 2024-08-19T02:19:26-07:00
draft = false
tags = ["AI", "machine-learning", "startup"]
+++

Post content in Markdown...
```

## Common Tags

**AI/ML**: AI, LLM, LLMs, machine-learning, synthetic-data, data-generation, chain-of-thought, chain-of-draft, reinforcement-learning

**Business**: startup, entrepreneurship, product-development, business, leadership, team-management, company-culture, hiring, scaling

## Working with Content

### Editing Posts
- All posts are in `dylan-blog/content/posts/`
- Use Markdown format with TOML frontmatter
- Images go in `dylan-blog/content/images/` and reference as `/images/filename.png`

### Adding Images
1. Place image in `dylan-blog/content/images/`
2. Reference in post: `![Alt text](/images/your-image.png)`

### Code Blocks
Use standard Markdown fenced code blocks with language hints:
````markdown
```python
def example():
    return "Hello"
```
````

## Configuration Reference

Main config: `dylan-blog/hugo.toml`

Key settings:
- `baseURL`: https://dylanler.github.io/
- `theme`: PaperMod (dark mode default)
- `paginate`: 10 posts per page
- `ShowReadingTime`: true
- `ShowToc`: true (table of contents)
- `ShowCodeCopyButtons`: true

## Draft Research Files

The `/drafts/` directory contains research notes that may become blog posts:
- `sft-training-diverse.txt` - SFT training dataset research
- `grpo-document.txt` - GRPO notes
- `CoD-deepresearch.txt` - Chain of Draft research
- `camera-movement-dataset.txt` - Video dataset notes

These are NOT automatically published.

## Important Rules

1. **NEVER edit files in `dylan-blog/public/`** - These are auto-generated
2. **NEVER commit with `draft = true`** unless intentionally saving work-in-progress
3. **Always test locally** with `hugo serve` before pushing
4. **Theme is a git submodule** - Update with `git submodule update --remote`

## Git Workflow

```bash
# Stage content changes
git add dylan-blog/content/posts/

# Commit with descriptive message
git commit -m "Add new blog post about X"

# Push to deploy
git push origin main
```

The GitHub Actions workflow (`.github/workflows/hugo.yml`) handles:
1. Installing Hugo v0.133.0 and Dart Sass
2. Checking out submodules (PaperMod theme)
3. Building the site with `hugo --minify`
4. Deploying to GitHub Pages

## Typical Tasks

### Task: Create a new blog post from draft
1. Read the draft file in `/drafts/`
2. Create new post: `cd dylan-blog && hugo new posts/post-name.md`
3. Copy and format content with proper Markdown
4. Add appropriate tags
5. Set `draft = false`
6. Test with `hugo serve`
7. Commit and push

### Task: Update existing post
1. Read the post in `dylan-blog/content/posts/`
2. Make edits directly to the .md file
3. Test with `hugo serve`
4. Commit and push

### Task: Add images to a post
1. Place image in `dylan-blog/content/images/`
2. Reference as `![Description](/images/filename.png)`
3. Commit both the image and the updated post

## Theme Features (PaperMod)

- **Dark mode**: Default enabled
- **Search**: Client-side with Fuse.js (JSON index at `/index.json`)
- **RSS**: Available at `/index.xml`
- **Social links**: GitHub, Twitter (configured in hugo.toml)
- **Reading time**: Auto-calculated
- **Table of contents**: Auto-generated from headings

## Troubleshooting

### Build fails locally
- Ensure Hugo v0.133.0+ is installed
- Run `git submodule update --init` to fetch PaperMod theme

### Post not appearing
- Check `draft = false` in frontmatter
- Verify date is not in the future
- Ensure file is in `dylan-blog/content/posts/`

### Images not loading
- Use absolute path from site root: `/images/filename.png`
- Ensure image is in `dylan-blog/content/images/`

### Pushing to repo
- Use cat ~/.ssh/config    
- Use personal github config to push

### Getting latest documents

**IMPORTANT**: Before writing experiment code or using external libraries, ALWAYS:

1. **Use Context7 MCP** to get latest documentation:
   ```
   # First resolve the library ID
   mcp__context7__resolve-library-id(libraryName="library-name", query="what you need")

   # Then query the docs
   mcp__context7__query-docs(libraryId="/org/project", query="specific question")
   ```

2. **Use WebSearch** to find latest model versions and APIs:
   ```
   WebSearch(query="latest Claude API models January 2026")
   WebSearch(query="OpenAI GPT-5 API documentation 2026")
   ```

3. **Check for latest versions** of:
   - LLM APIs (Anthropic, OpenAI, Google)
   - Python libraries (use `uv` for dependency management)
   - Framework versions

### Example: Finding latest LLM models
```
# Search for current models
WebSearch(query="Anthropic Claude latest models API 2026")
WebSearch(query="OpenAI GPT models API January 2026")

# Get uv documentation for scripts
mcp__context7__resolve-library-id(libraryName="uv", query="inline script dependencies")
mcp__context7__query-docs(libraryId="/astral-sh/uv", query="PEP 723 inline dependencies")
```

## Experiment Tools

The `/experiment-tools/` directory contains Python scripts for running LLM experiments.

### Using uv for Scripts

All experiment scripts use **uv with inline dependencies** (PEP 723). No virtual environment needed.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run any script directly
uv run experiment-tools/life_decision_eval.py
uv run experiment-tools/life_simulator.py --episodes 50
uv run experiment-tools/biography_extractor.py --person "Richard Feynman" --auto
```

### Script Format (PEP 723)
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40.0",
#   "openai>=1.50.0",
#   "rich>=13.0.0",
# ]
# ///

import anthropic
# ... rest of script
```

### Available Experiment Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `life_decision_eval.py` | MCQ evaluation of LLM decision-making | `uv run life_decision_eval.py --model claude-opus` |
| `life_simulator.py` | Monte Carlo life trajectory simulation | `uv run life_simulator.py --compare` |
| `biography_extractor.py` | Extract decision points from biographies | `uv run biography_extractor.py --all` |
| `value_function_compare.py` | Compare value functions across LLMs | `uv run value_function_compare.py --models claude-opus,gpt-5` |

### Environment Variables

Create `.env` in `experiment-tools/`:
```bash
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

