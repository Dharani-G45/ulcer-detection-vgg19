document.addEventListener(
    "DOMContentLoaded",
    () => {

        const uploadZone =
            document.getElementById(
                "uploadZone"
            );

        const imageInput =
            document.getElementById(
                "imageInput"
            );

        const uploadPlaceholder =
            document.getElementById(
                "uploadPlaceholder"
            );

        const previewContainer =
            document.getElementById(
                "previewContainer"
            );

        const imagePreview =
            document.getElementById(
                "imagePreview"
            );

        const fileInformation =
            document.getElementById(
                "fileInformation"
            );

        const fileName =
            document.getElementById(
                "fileName"
            );

        const removeImage =
            document.getElementById(
                "removeImage"
            );

        const predictionForm =
            document.getElementById(
                "predictionForm"
            );

        const predictButton =
            document.getElementById(
                "predictButton"
            );

        const predictButtonText =
            document.getElementById(
                "predictButtonText"
            );

        const mobileMenuButton =
            document.getElementById(
                "mobileMenuButton"
            );

        const navigation =
            document.getElementById(
                "navigation"
            );

        const navLinks =
            document.querySelectorAll(
                ".nav-link"
            );


        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/webp",
        ];


        const maximumFileSize =
            5 * 1024 * 1024;


        function displaySelectedImage(
            file
        ) {

            if (!file) {
                return;
            }


            if (
                !allowedTypes.includes(
                    file.type
                )
            ) {

                alert(
                    "Please select a JPG, PNG or WEBP image."
                );

                clearSelectedImage();

                return;
            }


            if (
                file.size >
                maximumFileSize
            ) {

                alert(
                    "The image must be 5 MB or smaller."
                );

                clearSelectedImage();

                return;
            }


            const reader =
                new FileReader();


            reader.addEventListener(
                "load",
                (event) => {

                    imagePreview.src =
                        event.target.result;

                    uploadPlaceholder.style.display =
                        "none";

                    previewContainer.classList.add(
                        "visible"
                    );

                    fileInformation.classList.add(
                        "visible"
                    );

                    fileName.textContent =
                        file.name;

                    predictButton.disabled =
                        false;
                }
            );


            reader.readAsDataURL(
                file
            );
        }


        function clearSelectedImage() {

            imageInput.value = "";

            imagePreview.src = "";

            previewContainer.classList.remove(
                "visible"
            );

            uploadPlaceholder.style.display =
                "block";

            fileInformation.classList.remove(
                "visible"
            );

            fileName.textContent =
                "No file selected";

            predictButton.disabled =
                true;

            predictButtonText.textContent =
                "Predict Ulcer Type";
        }


        uploadZone.addEventListener(
            "click",
            () => {

                imageInput.click();
            }
        );


        imageInput.addEventListener(
            "change",
            () => {

                displaySelectedImage(
                    imageInput.files[0]
                );
            }
        );


        removeImage.addEventListener(
            "click",
            (event) => {

                event.preventDefault();

                event.stopPropagation();

                clearSelectedImage();
            }
        );


        uploadZone.addEventListener(
            "dragover",
            (event) => {

                event.preventDefault();

                uploadZone.classList.add(
                    "drag-active"
                );
            }
        );


        uploadZone.addEventListener(
            "dragleave",
            () => {

                uploadZone.classList.remove(
                    "drag-active"
                );
            }
        );


        uploadZone.addEventListener(
            "drop",
            (event) => {

                event.preventDefault();

                uploadZone.classList.remove(
                    "drag-active"
                );


                const files =
                    event.dataTransfer.files;


                if (!files.length) {
                    return;
                }


                const file =
                    files[0];


                if (
                    !allowedTypes.includes(
                        file.type
                    )
                ) {

                    alert(
                        "Please select a JPG, PNG or WEBP image."
                    );

                    return;
                }


                if (
                    file.size >
                    maximumFileSize
                ) {

                    alert(
                        "The image must be 5 MB or smaller."
                    );

                    return;
                }


                const dataTransfer =
                    new DataTransfer();


                dataTransfer.items.add(
                    file
                );


                imageInput.files =
                    dataTransfer.files;


                displaySelectedImage(
                    file
                );
            }
        );


        predictionForm.addEventListener(
            "submit",
            () => {

                if (
                    !imageInput.files.length
                ) {

                    return;
                }


                predictButton.disabled =
                    true;

                predictButtonText.textContent =
                    "Analyzing Image...";
            }
        );


        mobileMenuButton.addEventListener(
            "click",
            () => {

                navigation.classList.toggle(
                    "open"
                );

                document.body.classList.toggle(
                    "menu-open"
                );
            }
        );


        navLinks.forEach(
            (link) => {

                link.addEventListener(
                    "click",
                    () => {

                        navigation.classList.remove(
                            "open"
                        );

                        document.body.classList.remove(
                            "menu-open"
                        );


                        navLinks.forEach(
                            (item) => {

                                item.classList.remove(
                                    "active"
                                );
                            }
                        );


                        link.classList.add(
                            "active"
                        );
                    }
                );
            }
        );


        /*
        If Django has rendered a previous result,
        the image preview may exist but the browser
        intentionally does not repopulate file inputs.

        Therefore another image must be chosen before
        another prediction can be submitted.
        */

        if (
            !imageInput.files.length
        ) {

            predictButton.disabled =
                true;
        }

    }
);