# Latents Visualizations - Vercel Deployment

This directory contains the static site deployed to Vercel, showcasing interactive research visualizations and example dashboards.

## Structure

```
public/
├── index.html              # Landing page
├── research/               # Research experiment visualizations
│   └── confound-analysis.html
└── examples/               # Example demonstrations
    ├── interactive-sandbox.html
    ├── complete-results.html
    ├── results-dashboard.html
    └── temporal-explorer.html
```

## Deployment

### Deploy to Vercel

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy**:
   ```bash
   # From project root
   vercel deploy

   # For production
   vercel deploy --prod
   ```

3. **Link to GitHub** (recommended):
   - Connect your GitHub repository in Vercel dashboard
   - Automatic deployments on push to main branch

### Manual Deployment

If you prefer to use the Vercel web interface:
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Vercel will auto-detect the configuration from `vercel.json`
4. Deploy!

## Adding New Visualizations

When you generate new Plotly visualizations:

```bash
# Copy research results
cp research/results/your_new_plot.html public/research/your-new-plot.html

# Copy examples
cp examples/visualizations/your_example.html public/examples/your-example.html

# Update index.html to add links to new visualizations
# Then commit and push (auto-deploys if linked to GitHub)
```

Or use the update script:

```bash
./scripts/update-visualizations.sh
```

## Configuration

### vercel.json

- Configures static file serving from `public/` directory
- Sets security headers
- Configures caching (1 hour for HTML files)

### .vercelignore

- Excludes Python code, data files, and large binary files
- Only deploys the `public/` directory
- Keeps deployment size minimal

## Environment Variables

No environment variables needed - this is a static site deployment.

## Custom Domain

To add a custom domain:
1. Go to Vercel dashboard → Your Project → Settings → Domains
2. Add your domain (e.g., `results.latents.ai`)
3. Update DNS records as instructed

## Analytics

Vercel provides built-in analytics:
- Enable in Project Settings → Analytics
- Track page views, performance, and visitor metrics

## Updating Content

### Automatic (GitHub Integration)

When linked to GitHub, every push to `main` triggers a deployment:

```bash
# Generate new visualization
python research/experiments/new_experiment.py

# Copy to public/
cp research/results/new_viz.html public/research/

# Update index.html with link
# ... edit public/index.html ...

# Commit and push
git add public/
git commit -m "Add new visualization: new_viz"
git push origin main

# Vercel automatically deploys!
```

### Manual

```bash
vercel deploy --prod
```

## Access

After deployment, your site will be available at:
- Development: `https://latents-<random>.vercel.app`
- Production: `https://latents.vercel.app` (or your custom domain)

## Notes

- All visualizations are self-contained HTML files (Plotly standalone)
- No backend required - pure static hosting
- Fast global CDN delivery via Vercel Edge Network
- Automatic HTTPS
- Preview deployments for pull requests
