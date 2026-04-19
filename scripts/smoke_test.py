import requests


def main():
    base = "http://127.0.0.1:8000"

    print(requests.get(f"{base}/").json())
    print(requests.get(f"{base}/health").json())
    print(requests.get(f"{base}/ready").json())
    print(requests.get(f"{base}/metrics").json())


if __name__ == "__main__":
    main()