import logging

import jaxtyping
import torch

logger = logging.getLogger(__name__)


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.proj = None

    def fit(self, X: torch.Tensor, offset: int = 0) -> None:
        # Center the data
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = torch.cov(X_centered.T)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.proj = eigenvectors[:, sorted_indices][:, : self.n_components]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # Center the data
        X_centered = X - self.mean
        return X_centered @ self.proj

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)  # Return the transformed data


def get_class_map(labels: list[str | int]):
    return {label: idx for idx, label in enumerate(set(labels))}


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, n_classes, class_map=None, name="LinearProbe"):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, n_classes, bias=False)
        torch.nn.init.kaiming_normal_(self.linear.weight)
        self.class_head = torch.nn.Softmax(dim=-1)
        self.class_map = class_map
        self.name = name

    def forward(self, x):
        logits = self.linear(x)
        proba = self.class_head(logits)
        return proba

    @torch.inference_mode()
    def predict(
        self, x: jaxtyping.Float32[torch.Tensor, "batch features"], map_labels=True
    ):
        self.eval()
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.stack(x).to(self.device)
        outputs = self(x)
        _, predicted = torch.max(outputs, dim=-1)
        if map_labels:
            return [list(self.class_map.keys())[i] for i in predicted.tolist()]
        return predicted

    @torch.inference_mode()
    def validate(
        self,
        x: jaxtyping.Float32[torch.Tensor, "batch features"],
        y: list[str | int],
        batch_size=None,
        return_confusion_matrix=False,
    ):
        self.eval()
        y = torch.tensor([self.class_map[label] for label in y]).to(self.device)
        batch_size = batch_size or len(x)
        if return_confusion_matrix:
            confusion_matrix = torch.zeros(len(self.class_map), len(self.class_map))

        correct = 0
        total = 0
        for i in range(0, len(x), batch_size):
            batch_x = x[i : i + batch_size]
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(self.device)
            else:
                batch_x = torch.stack(batch_x).to(self.device)
            batch_y = y[i : i + batch_size]
            outputs = self(batch_x)
            _, predicted = torch.max(outputs, dim=-1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            if return_confusion_matrix:
                for t, p in zip(batch_y, predicted):
                    confusion_matrix[t, p] += 1

        if return_confusion_matrix:
            return correct / total, confusion_matrix

        return correct / total

    def save(self, path):
        torch.save(
            {
                "name": self.name,
                "class_map": self.class_map,
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def from_pretained(path, device="cuda" if torch.cuda.is_available() else "cpu"):
        pt = torch.load(path)
        shape = pt["model_state_dict"]["linear.weight"].shape
        probe = LinearProbe(input_dim=shape[1], n_classes=shape[0])
        probe.name = pt["name"]
        probe.class_map = pt["class_map"]
        probe.load_state_dict(pt["model_state_dict"])
        probe.to(device)

        return probe

    @staticmethod
    def from_data(
        acts: jaxtyping.Float32[torch.Tensor, "batch features"],
        labels: list[str | int],
        lr=0.0001,
        weight_decay=0.01,
        epochs=1000,
        batch_size=None,
        log_steps=100,
        device: torch.device = "cuda",
        name: str = "LinearProbe",
        validation_set: tuple[
            jaxtyping.Float32[torch.Tensor, "batch features"], list[str | int]
        ] = None,
        print_logs=True,
    ):
        # acts = acts.to(device)
        class_map = get_class_map(labels)
        print(class_map)

        probe = LinearProbe(
            input_dim=acts.shape[1], n_classes=len(class_map), name=name
        ).to(device)
        probe.class_map = class_map

        labels = torch.tensor([class_map[label] for label in labels]).to(device)

        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        batch_size = batch_size or len(acts)
        step = 0
        for epoch in range(epochs):
            probe.train()
            for i in range(0, len(acts), batch_size):
                if isinstance(acts, torch.Tensor):
                    batch_acts = acts[i : i + batch_size].to(device)
                else:
                    batch_acts = torch.stack(acts[i : i + batch_size]).to(device)

                batch_labels = labels[i : i + batch_size]

                optimizer.zero_grad()
                outputs = probe(batch_acts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                step += 1

                if print_logs and step % log_steps == 0:
                    log_msg = f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}"
                    if validation_set is not None:
                        val_x, val_y = validation_set
                        val_acc = probe.validate(val_x, val_y, batch_size)
                        log_msg += f", Validation Accuracy: {val_acc:.4f}"

                    logger.debug(log_msg)

        probe.eval()
        if validation_set is not None:
            val_x, val_y = validation_set
            val_acc = probe.validate(val_x, val_y, batch_size)
            logger.info(f"{probe.name} validation accuracy: {val_acc:.4f}")
        return probe
