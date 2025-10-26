#!/bin/bash

# Update visualizations for Vercel deployment
# Copies latest visualizations from research/results and examples to public/

set -e

echo "Updating visualizations for Vercel deployment..."

# Create directories if they don't exist
mkdir -p public/research
mkdir -p public/examples

# Copy research results
echo "Copying research visualizations..."
if [ -f "research/results/pca_temporal_style_interactive.html" ]; then
    cp research/results/pca_temporal_style_interactive.html public/research/confound-analysis.html
    echo "✓ Copied confound analysis PCA plot"
fi

# Copy any other research HTML files
for file in research/results/*.html; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Skip if already copied
        if [ "$filename" != "pca_temporal_style_interactive.html" ]; then
            # Convert underscores to hyphens for URL-friendly names
            newname=$(echo "$filename" | sed 's/_/-/g')
            cp "$file" "public/research/$newname"
            echo "✓ Copied $filename → $newname"
        fi
    fi
done

# Copy example visualizations
echo "Copying example visualizations..."
if [ -f "examples/visualizations/interactive_sandbox.html" ]; then
    cp examples/visualizations/interactive_sandbox.html public/examples/interactive-sandbox.html
    echo "✓ Copied interactive sandbox"
fi

if [ -f "examples/visualizations/complete_results.html" ]; then
    cp examples/visualizations/complete_results.html public/examples/complete-results.html
    echo "✓ Copied complete results dashboard"
fi

if [ -f "examples/visualizations/results_dashboard.html" ]; then
    cp examples/visualizations/results_dashboard.html public/examples/results-dashboard.html
    echo "✓ Copied results dashboard"
fi

if [ -f "examples/visualizations/temporal_explorer.html" ]; then
    cp examples/visualizations/temporal_explorer.html public/examples/temporal-explorer.html
    echo "✓ Copied temporal explorer"
fi

echo ""
echo "✅ Visualization update complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit: git add public/ && git commit -m 'Update visualizations'"
echo "3. Push: git push origin main"
echo "4. Vercel will auto-deploy if linked to GitHub"
echo ""
echo "Or deploy manually:"
echo "  vercel deploy --prod"
