import base64

from django.shortcuts import render

from .forms import ImageUploadForm
from .services.predictor import (
    predict_endoscopy_image,
)


def home(request):
    """
    Main ulcer-detection application.

    GET:
        Display upload interface.

    POST:
        Validate uploaded image.
        Run the fine-tuned VGG19 model.
        Display prediction results.
    """

    form = ImageUploadForm()

    context = {
        "form": form,
        "prediction": None,
        "confidence": None,
        "probabilities": None,
        "preview_data_url": None,
        "uploaded_filename": None,
        "error_message": None,
    }

    if request.method == "POST":

        form = ImageUploadForm(
            request.POST,
            request.FILES,
        )

        context["form"] = form

        if form.is_valid():

            uploaded_image = form.cleaned_data["image"]

            try:

                uploaded_image.seek(0)

                image_bytes = uploaded_image.read()

                uploaded_image.seek(0)

                # --------------------------------------------
                # Create browser preview without saving
                # the upload permanently to disk.
                # --------------------------------------------

                encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                content_type = uploaded_image.content_type or "image/jpeg"

                context["preview_data_url"] = (
                    f"data:{content_type};" f"base64,{encoded_image}"
                )

                context["uploaded_filename"] = uploaded_image.name

                # --------------------------------------------
                # REAL AI PREDICTION
                # --------------------------------------------

                prediction_result = predict_endoscopy_image(image_bytes)

                context["prediction"] = prediction_result["prediction_label"]

                context["confidence"] = prediction_result["confidence"]

                context["probabilities"] = prediction_result["probabilities"]

            except Exception as error:

                print(
                    "Prediction error:",
                    error,
                )

                context["error_message"] = (
                    "The image could not be analyzed. "
                    "Please try another valid endoscopy image."
                )

    return render(
        request,
        "detector/upload.html",
        context,
    )
