const submitBtn = document.getElementById('submitBtn');
const inputField = document.getElementById('inputField');
const output = document.getElementById('output');

submitBtn.addEventListener('click', () => {
  const inputText = inputField.value;
  if (inputText.trim() !== '') {
    output.innerHTML += `<p class="user-text">${inputText}</p>`;
    inputField.value = '';

    fetch('/chat', {
      method: 'POST',
      body: JSON.stringify({
        message: inputText
      }),
      headers: {
        'Content-Type': 'application/json'
      }
    })
      .then(response => response.json())
      .then(data => {
        const reply = data['response'];
        output.innerHTML += `<p class="bot-text">${reply}</p>`;
      });
  }
});

inputField.addEventListener('keyup', (event) => {
  if (event.keyCode === 13) {
    event.preventDefault();
    submitBtn.click();
  }
});
