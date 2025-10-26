#!/bin/bash

# GCP Instance Manager for Temporal Grounding Experiment

PROJECT="new-one-82feb"
ZONE="us-central1-a"
INSTANCE="temporal-gpt2-experiment"
BUCKET="gs://temporal-grounding-gpt2-82feb"

function create_instance() {
    echo "Creating GCP instance..."
    gcloud compute instances create $INSTANCE \
        --project=$PROJECT \
        --zone=$ZONE \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=50GB \
        --maintenance-policy=TERMINATE \
        --preemptible \
        --scopes=cloud-platform

    echo ""
    echo "✓ Instance created!"
    echo "Wait 1-2 minutes for startup, then run: $0 ssh"
}

function ssh_instance() {
    gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE
}

function delete_instance() {
    echo "Deleting instance..."
    gcloud compute instances delete $INSTANCE --project=$PROJECT --zone=$ZONE --quiet
    echo "✓ Instance deleted!"
}

function status() {
    echo "Instance status:"
    gcloud compute instances list --project=$PROJECT --filter="name=$INSTANCE"
}

function download_results() {
    echo "Downloading all results from GCS bucket..."
    gsutil -m rsync -r $BUCKET/results/ results/
    gsutil -m rsync -r $BUCKET/probes/ probes/
    gsutil -m rsync -r $BUCKET/activations/ activations/
    echo "✓ Results downloaded!"
}

function upload_code() {
    echo "Uploading code to GCS bucket..."
    gsutil -m rsync -r -x "venv/.*" . $BUCKET/code/
    echo "✓ Code uploaded!"
}

function view_bucket() {
    echo "Contents of GCS bucket:"
    gsutil ls -r $BUCKET/
}

function help() {
    echo "GCP Manager for Temporal Grounding Experiment"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  create        Create GCP instance with GPU"
    echo "  ssh           SSH into instance"
    echo "  delete        Delete instance (IMPORTANT: do this when done!)"
    echo "  status        Check instance status"
    echo "  download      Download all results from GCS"
    echo "  upload        Upload code to GCS"
    echo "  bucket        View GCS bucket contents"
    echo "  help          Show this help"
    echo ""
    echo "Example workflow:"
    echo "  1. $0 upload         # Upload latest code"
    echo "  2. $0 create         # Create instance"
    echo "  3. $0 ssh            # SSH and run experiment"
    echo "  4. $0 download       # Download results"
    echo "  5. $0 delete         # Delete instance to stop charges"
}

# Main
case "$1" in
    create)
        create_instance
        ;;
    ssh)
        ssh_instance
        ;;
    delete)
        delete_instance
        ;;
    status)
        status
        ;;
    download)
        download_results
        ;;
    upload)
        upload_code
        ;;
    bucket)
        view_bucket
        ;;
    help|*)
        help
        ;;
esac
