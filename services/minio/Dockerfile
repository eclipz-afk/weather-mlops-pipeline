# The original public MinIO image was removed from Quay/Docker Hub.
# Build the same release from its official source without changing the volume.
FROM golang:1.24.2-bookworm AS build
ARG MINIO_RELEASE=RELEASE.2025-07-23T15-54-02Z
RUN git clone --depth 1 --branch "${MINIO_RELEASE}" https://github.com/minio/minio.git /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -p 2 -trimpath -ldflags="-s -w" -o /out/minio .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/minio /usr/local/bin/minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
