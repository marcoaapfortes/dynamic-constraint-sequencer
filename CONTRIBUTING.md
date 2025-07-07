# Contributing to Dynamic Constraint Sequencer

Thank you for considering contributing! 🎉 Your help is what makes this project thrive.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of data structures and algorithms
- Familiarity with pandas and numpy (helpful but not required)

### Setting Up Your Development Environment

1. **Fork the repository**
   - Click the "Fork" button on the GitHub repository page
   - This creates your own copy of the repository

2. **Clone your fork**
   ```bash
   git clone https://github.com/marcoaapfortes/dynamic-constraint-sequencer.git
   cd dynamic-constraint-sequencer
   ```

3. **Set up a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install optional dependencies for full functionality**
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Verify your setup**
   ```bash
   python -m py_compile dcs.py
   python dcs.py  # Run the benchmark
   ```

7. **Set up the upstream remote**
   ```bash
   git remote add upstream https://github.com/marcoaapfortes/dynamic-constraint-sequencer.git
   ```

## 📖 Development Guidelines

### Code Style
- Follow **PEP 8** Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all classes and functions
- Keep functions focused and small (ideally < 50 lines)
- Use type hints where appropriate

### Code Formatting
We use `black` for code formatting:
```bash
pip install black
black .
```

### Documentation
- Update docstrings for any new or modified functions
- Add inline comments for complex logic
- Update README.md if you add new features
- Include examples in docstrings when helpful

### Testing
- Write unit tests for new features
- Ensure existing tests still pass
- Test energy monitoring features if applicable
- Test with different dataset sizes and sparsity levels

```bash
# Run basic syntax check
python -m py_compile dcs.py

# Run the benchmark to test functionality
python dcs.py
```

## 🌱 Making a Pull Request (PR)

### Before You Start
1. **Check existing issues** to see if someone is already working on your idea
2. **Open an issue** to discuss major changes before implementing them
3. **Look for "good first issue"** labels if you're new to the project

### Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b bugfix/fix-energy-calculation
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Follow the existing code style
   - Add tests if applicable

3. **Test your changes**
   ```bash
   python -m py_compile dcs.py
   python dcs.py  # Run benchmark
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add sparsity optimization for categorical data"
   ```

5. **Keep your branch up to date**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Open a Pull Request**
   - Use the PR template
   - Provide a clear description of your changes
   - Link any related issues
   - Include performance impact if applicable

### Commit Message Guidelines
We follow [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `perf:` performance improvements
- `refactor:` code refactoring
- `test:` adding or updating tests
- `chore:` maintenance tasks

Examples:
```
feat: add reinforcement learning for constraint ordering
fix: resolve memory leak in energy monitoring
docs: update installation instructions
perf: optimize sparsity detection algorithm
```

## 🎯 Types of Contributions

### 🐛 Bug Fixes
- Fix energy calculation errors
- Resolve memory leaks
- Correct algorithm logic issues
- Improve error handling

### ✨ New Features
- New optimization algorithms
- Enhanced query parsing
- Additional energy monitoring metrics
- Edge AI optimizations
- Support for new data types

### 📚 Documentation
- Improve README.md
- Add code examples
- Create tutorials
- Fix typos and grammar

### 🔧 Performance Improvements
- Optimize algorithms for speed
- Reduce memory usage
- Improve energy efficiency
- Better sparsity exploitation

### 🧪 Testing
- Add unit tests
- Create integration tests
- Benchmark new features
- Test edge cases

## 🏷 Issue Labels

When contributing, look for these labels:

- `good first issue`: Perfect for newcomers
- `help wanted`: We need community help
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to docs
- `performance`: Speed or efficiency improvements
- `energy-efficiency`: Related to power consumption
- `edge-ai`: Edge computing optimizations

## 📊 Performance Guidelines

When making changes that could affect performance:

1. **Measure before and after** using the built-in benchmark
2. **Test with different dataset sizes** (1K, 10K, 50K+ records)
3. **Monitor energy consumption** if your changes affect algorithms
4. **Document performance impact** in your PR

### Benchmark Results Format
```
Query: "Find cheapest red phones under $300"
Traditional: 45ms, 200K ops, 50 cost units
Enhanced: 15ms, 80K ops, 20 cost units, 25J energy
Improvement: 3x faster, 60% fewer ops, 60% lower cost
```

## 🔍 Code Review Process

1. **All PRs require review** before merging
2. **Be responsive to feedback** and make requested changes
3. **Tests must pass** (syntax check and benchmark)
4. **Documentation must be updated** for new features
5. **Performance impact must be documented** for algorithm changes

### What Reviewers Look For
- Code quality and readability
- Proper error handling
- Performance impact
- Test coverage
- Documentation updates
- Adherence to project standards

## 🌍 Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Be Helpful
- Help newcomers get started
- Share knowledge and best practices
- Provide constructive feedback
- Celebrate others' contributions

## 💬 Getting Help

### Questions?
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Code Review**: Ask questions in PR comments
- **Documentation**: Check README.md and code comments

### Stuck?
- Look at existing code for examples
- Check the benchmark implementation in `dcs.py`
- Review recent PRs for similar changes
- Ask questions in your PR or issue

## 🎉 Recognition

Contributors will be:
- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Credited in documentation they help improve

## 📝 License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

Thank you for contributing to Dynamic Constraint Sequencer! Every contribution, no matter how small, helps make this project better for everyone. 🚀