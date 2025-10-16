import daft

from jotunn.components.text.refusals import RefusalClassifier

df = daft.from_pydict(
    {
        "text": [
            "My mother told me",
            "Someday I will buy",
            "Galleys with good oars",
            "Sail to distant shores",
            "Sorry, I cannot provide a caption for this image.",
            "I cannot provide assistance with illegal activities.",
        ],
    }
)

classifier = RefusalClassifier(
    input_column="text",
    batch_size=6,
    concurrency=1,
    num_cpus=6,
    num_gpus=1,
)
df = classifier(df)
df.show()
