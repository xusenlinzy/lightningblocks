import random

from lightningblocks.dataaug.base_augment import BaseAugment

__all__ = ['WordDelete']


class WordDelete(BaseAugment):
    """
    WordDelete is a word-level deletion data augmentation strategy.
    Args:
        create_n (int):
            Number of augmented sequences.
        aug_n (int):
            Number of augmented words in sequences.
        aug_percent (int):
            Percentage of augmented words in sequences.
        aug_min (int):
            Minimum number of augmented words in sequences.
        aug_max (int):
            Maximum number of augmented words in sequences.
    """

    def __init__(
        self,
        create_n=1,
        aug_n=None,
        aug_percent=0.02,
        aug_min=1,
        aug_max=10
    ) -> None:
        super().__init__(create_n=create_n,
                         aug_n=aug_n,
                         aug_percent=aug_percent,
                         aug_min=aug_min,
                         aug_max=aug_max)

    def _augment(self, sequence):

        seq_tokens = self.tokenizer.cut(sequence)
        aug_indexes = self._skip_words(seq_tokens)
        aug_n = self._get_aug_n(len(seq_tokens), len(aug_indexes))

        t = 0
        sentences = []
        if aug_n == 0:
            return sentences
        while t < self.create_n * self.loop and len(sentences) < self.create_n:
            t += 1
            idxes = random.sample(aug_indexes, aug_n)
            sentence = ''.join(seq_tokens[idx] for idx in range(len(seq_tokens)) if idx not in idxes)

            if sentence not in sentences:
                sentences.append(sentence)
        return sentences

    def _skip_words(self, seq_tokens):
        """Skip words. We can rewrite function to skip specify words."""
        return [i for i, seq_token in enumerate(seq_tokens) if seq_token not in self.stop_words and not seq_token.isdigit() and not seq_token.encode('UTF-8').isalpha()]
