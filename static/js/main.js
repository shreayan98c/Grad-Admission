console.log("Connected to JS!")

function validate() {
	var gre = document.getElementById('gre').value;
	if(gre<260 || gre>340)
	{
		return false;
	}

	return true;
}