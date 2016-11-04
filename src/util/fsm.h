
template<typename T>
class FSM
{
public:
	FSM() : _cur_state(-1) {}
	virtual ~FSM() {};

	// Virtual function to define a precondition of a state
	virtual void BeginState( T state ) {}
	virtual void UpdateState( T state ) {}
	// Virtual function to define a postcondition of a state
	virtual void EndState( T state ) {}

	void SetState( T state )
	{
		EndState( (T)_cur_state );
		_cur_state = state;
		BeginState( (T)_cur_state );
	}

	void UpdateFSM( )
	{
		if( _cur_state != -1 )
		{
			UpdateState( (T)_cur_state );
		}
	}

	T GetState() { return (T)_cur_state; }

private:
	int _cur_state;
};
