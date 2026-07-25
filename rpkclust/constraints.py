import numpy as np


class ClusteringConstraints:

    @staticmethod
    def _hamming_distance(a, b):

        length = min(len(a), len(b))

        dist = 0

        for i in range(length):
            dist += bin(a[i] ^ b[i]).count("1")

        dist += abs(len(a)-len(b))*8

        return dist
        
    @staticmethod
    def message_similarity(labels, X):

        clusters = {}

        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)


        scores = []

        for indexes in clusters.values():

            if len(indexes) < 2:
                continue


            distances = []

            for i in range(len(indexes)):
                for j in range(i+1, len(indexes)):

                    d = ClusteringConstraints._hamming_distance(
                        X[indexes[i]],
                        X[indexes[j]]
                    )

                    distances.append(d)


            if distances:

                max_dist = max(distances)

                if max_dist > 0:
                    similarity = 1 - (
                        np.mean(distances) / max_dist
                    )
                else:
                    similarity = 1.0

                scores.append(similarity)


        if not scores:
            return 0.0


        return float(np.mean(scores))


    @staticmethod
    def remote_coupling(labels, X):

        clusters = {}

        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)


        cluster_scores = []


        for indexes in clusters.values():

            if len(indexes) < 2:
                continue


            lengths = [
                len(X[i])
                for i in indexes
            ]


            variance = np.var(lengths)


            score = 1 / (1 + variance)

            cluster_scores.append(score)


        if not cluster_scores:
            return 0.0


        return float(np.mean(cluster_scores))


    @staticmethod
    def structural_consistency(labels, candidate):

        values = candidate["values"]


        presence = sum(
            1 for v in values
            if v is not None
        ) / len(values)


        if candidate["type"] == "FOR":

            offset_score = 1.0

        else:

            offset = candidate.get(
                "offset",
                0
            )

            offset_score = 1 / (1 + offset)


        return float(
            0.5 * presence +
            0.5 * offset_score
        )


    @staticmethod
    def dimensional_constraint(labels):

        n = len(labels)

        k = len(
            np.unique(labels)
        )


        if k <= 1:
            return 0.0


        ratio = k / n


        # sweet spot
        score = 1 - abs(
            ratio - 0.1
        )


        return float(
            np.clip(score,0,1)
        )