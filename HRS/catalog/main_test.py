import sys
def main():
    print(sys.argv)
    for arg in sys.argv[1:]:
        print(arg)

if __name__ == "__main__":
    main()