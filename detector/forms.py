from django import forms

MAXIMUM_IMAGE_SIZE = 5 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class ImageUploadForm(forms.Form):

    image = forms.ImageField(
        label="Endoscopy Image",
        required=True,
        widget=forms.ClearableFileInput(
            attrs={
                "id": "imageInput",
                "class": "image-input",
                "accept": (".jpg,.jpeg,.png,.webp"),
            }
        ),
    )

    def clean_image(self):

        image = self.cleaned_data["image"]

        if image.size > MAXIMUM_IMAGE_SIZE:

            raise forms.ValidationError("Image size must be 5 MB or smaller.")

        content_type = getattr(
            image,
            "content_type",
            None,
        )

        if content_type and content_type not in ALLOWED_CONTENT_TYPES:

            raise forms.ValidationError("Only JPG, PNG and WEBP images are supported.")

        return image
